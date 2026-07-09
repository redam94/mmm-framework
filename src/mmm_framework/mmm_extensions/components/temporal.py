"""Trend + seasonality components for the extension models.

The extension family (NestedMMM / MultivariateMMM / CombinedMMM) historically
built no baseline dynamics — every mean was ``intercept + media (+ mediation /
cross-effects)`` only, so a real trend or seasonal pattern leaked into the media
coefficients. These helpers add lightweight, always-identified additive terms on
the **standardized-outcome scale** (the scale the extension likelihoods fit on),
reusing the SAME Fourier basis as the core :class:`BayesianMMM` so a nested
model's seasonality means the same thing as a plain MMM's.

They are deliberately self-contained: the term is a numpy design matrix (a
data-fixed constant — normalized time for the trend, sin/cos harmonics for
seasonality) multiplied by a small RV coefficient vector, so nothing here is a
parameter-independent ("dead") Deterministic and the Pathfinder ``_anchored_det``
invariant is unaffected. Both return ``None`` when the component is off, so the
caller adds ``x if x is not None else 0`` to its mean.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pymc as pm

if TYPE_CHECKING:
    import pytensor.tensor as pt

from ...transforms.seasonality import create_fourier_features


def _index_median_days(index: Any, n_obs: int) -> float | None:
    """Median spacing of a datetime index in days, or None for a non-datetime
    (RangeIndex / integer) index — seasonality needs real dates to fix its
    period. NB ``pd.DatetimeIndex(RangeIndex(...))`` would *succeed* (integers →
    nanosecond timestamps), so we must reject a non-datetime dtype explicitly."""
    is_dt = isinstance(index, pd.DatetimeIndex) or (
        hasattr(index, "dtype")
        and pd.api.types.is_datetime64_any_dtype(getattr(index, "dtype", None))
    )
    if not is_dt:
        return None
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return None
    deltas = np.diff(idx.view("int64")) / 1e9 / 86400.0  # ns -> days
    med = float(np.median(deltas))
    return med if med > 0 else None


def build_trend_contribution(
    prefix: str,
    n_obs: int,
    trend_config: Any,
) -> "pt.TensorVariable | None":
    """A standardized-scale additive trend term, or None when trend is off.

    Supports the full set of core trend families — ``linear``, ``piecewise``
    (Prophet-style changepoints), ``spline`` (random-walk-smoothed B-spline), and
    ``gaussian_process`` (HSGP) — reusing the SAME basis builders
    (``create_bspline_basis`` / ``create_piecewise_trend_matrix``) and HSGP that
    the core :class:`BayesianMMM` uses, so an extension model's trend means the
    same thing as a plain MMM's. Priors come from the ``TrendConfig`` fields.

    Every non-linear family is **zero-centered** (``trend - trend.mean()``) so its
    level is absorbed by the model intercept (``alpha_y`` / per-outcome ``alpha``)
    rather than fighting it — the identifiability guard. ``trend_config is None``
    or type ``none`` → no term (byte-identical to the historical extension graph).

    Must be called inside a ``pm.Model`` context; ``prefix`` namespaces the RVs
    (per-outcome for the multi-outcome models).
    """
    if trend_config is None:
        return None
    ttype = getattr(trend_config, "type", trend_config)
    ttype = str(getattr(ttype, "value", ttype)).lower()
    if ttype in ("none", "no_trend", ""):
        return None
    import pytensor.tensor as pt

    tc = trend_config
    t = np.linspace(0.0, 1.0, n_obs).astype(float)
    t_t = pt.as_tensor_variable(t)

    if ttype == "linear":
        slope = pm.Normal(
            f"{prefix}trend_slope",
            mu=float(getattr(tc, "growth_prior_mu", 0.0)),
            sigma=float(getattr(tc, "growth_prior_sigma", 0.5)),
        )
        return slope * t_t

    if ttype == "piecewise":
        from ...transforms.trend import create_piecewise_trend_matrix

        s, A = create_piecewise_trend_matrix(
            t,
            n_changepoints=int(getattr(tc, "n_changepoints", 10)),
            changepoint_range=float(getattr(tc, "changepoint_range", 0.8)),
        )
        k = pm.Normal(
            f"{prefix}trend_k",
            mu=float(getattr(tc, "growth_prior_mu", 0.0)),
            sigma=float(getattr(tc, "growth_prior_sigma", 0.5)),
        )
        delta = pm.Laplace(
            f"{prefix}trend_delta",
            mu=0.0,
            b=float(getattr(tc, "changepoint_prior_scale", 0.5)),
            shape=len(s),
        )
        A_t = pt.as_tensor_variable(A)
        # Prophet slope-change parameterization: each delta_j adjusts the growth
        # rate from changepoint s_j; the continuity term A @ (-s*delta) keeps the
        # trend continuous. The scalar Prophet offset `m` is OMITTED here: we
        # zero-center below (the level goes to the model intercept), which would
        # make a free `m` an unidentified dead parameter (it cancels exactly). The
        # continuity term is per-obs, so it survives centering.
        slope = k + pt.dot(A_t, delta)
        offset = pt.dot(A_t, -pt.as_tensor_variable(s) * delta)
        trend = slope * t_t + offset
        return trend - trend.mean()

    if ttype == "spline":
        from ...transforms.trend import create_bspline_basis

        basis = create_bspline_basis(
            t,
            n_knots=int(getattr(tc, "n_knots", 10)),
            degree=int(getattr(tc, "spline_degree", 3)),
        ).astype(float)
        coef_raw = pm.Normal(
            f"{prefix}spline_coef_raw", mu=0.0, sigma=1.0, shape=basis.shape[1]
        )
        scale = pm.HalfNormal(
            f"{prefix}spline_scale", sigma=float(getattr(tc, "spline_prior_sigma", 1.0))
        )
        # Random-walk-smoothed coefficients (cumsum), same as the core spline.
        coef = scale * pt.cumsum(coef_raw)
        trend = pt.dot(pt.as_tensor_variable(basis), coef)
        return trend - trend.mean()

    if ttype in ("gaussian_process", "gp"):
        import pymc.gp as gp_module

        # HSGP under MAP is unstable (the lengthscale is weakly identified from a
        # point estimate) — surface it so an approximate fit is read with caution.
        warnings.warn(
            "GP trend uses an HSGP approximation whose lengthscale is weakly "
            "identified under MAP/ADVI; prefer NUTS for a GP-trend extension fit.",
            UserWarning,
            stacklevel=2,
        )
        ls = pm.LogNormal(
            f"{prefix}gp_lengthscale",
            mu=float(np.log(float(getattr(tc, "gp_lengthscale_prior_mu", 0.3)))),
            sigma=float(getattr(tc, "gp_lengthscale_prior_sigma", 0.2)),
        )
        eta = pm.HalfNormal(
            f"{prefix}gp_amplitude",
            sigma=float(getattr(tc, "gp_amplitude_prior_sigma", 0.5)),
        )
        cov_func = eta**2 * gp_module.cov.Matern32(input_dim=1, ls=ls)
        gp = gp_module.HSGP(
            m=[int(getattr(tc, "gp_n_basis", 20))],
            c=float(getattr(tc, "gp_c", 1.5)),
            cov_func=cov_func,
        )
        t_gp = np.linspace(-1.0, 1.0, n_obs).reshape(-1, 1)
        trend = gp.prior(f"{prefix}trend_gp", X=t_gp)
        return trend - trend.mean()

    # Unknown type → fall back to linear (defensive; the spec validates the enum).
    warnings.warn(
        f"Unknown trend type '{ttype}'; using a linear trend.",
        UserWarning,
        stacklevel=2,
    )
    slope = pm.Normal(f"{prefix}trend_slope", mu=0.0, sigma=0.5)
    return slope * t_t


def build_seasonality_contribution(
    prefix: str,
    index: Any,
    n_obs: int,
    seasonality_config: Any,
    prior_sigma: float = 1.0,
) -> "pt.TensorVariable | None":
    """A standardized-scale additive Fourier seasonality term, or None.

    Periods are derived from the datetime index spacing (weekly data → yearly
    period ~52, etc.), mirroring the core model's frequency→period logic, so the
    component is comparable across models. Harmonics above the Nyquist limit are
    clamped. Returns None when: no config, all orders 0, or the index is not
    datetime (no way to fix a period) — the last case warns.
    """
    if seasonality_config is None:
        return None
    orders = {
        c: int(getattr(seasonality_config, c, 0) or 0)
        for c in ("yearly", "monthly", "weekly")
    }
    if not any(v > 0 for v in orders.values()):
        return None

    median_days = _index_median_days(index, n_obs)
    if median_days is None:
        warnings.warn(
            "Seasonality requested but the model has no datetime index to fix a "
            "period; skipping the seasonality component.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Observations per natural period, from the median inter-observation spacing.
    period_yearly = 365.25 / median_days
    component_periods = {
        "yearly": period_yearly,
        "monthly": period_yearly / 12.0,
        "weekly": period_yearly / 52.0,
    }

    import pytensor.tensor as pt

    t = np.arange(n_obs)
    feature_blocks: list[np.ndarray] = []
    for component in ("yearly", "monthly", "weekly"):
        order = orders[component]
        if order <= 0:
            continue
        period = component_periods[component]
        if period < 2.0:  # Nyquist: cannot resolve a period below 2 observations
            warnings.warn(
                f"{component} seasonality needs >= 2 observations per period "
                f"(got {period:.2f}); skipping this component.",
                UserWarning,
                stacklevel=2,
            )
            continue
        max_order = max(1, int(period / 2))
        if order > max_order:
            warnings.warn(
                f"{component} seasonality order {order} exceeds the max "
                f"resolvable order {max_order} for period {period:.2f}; clamping.",
                UserWarning,
                stacklevel=2,
            )
            order = max_order
        feats = create_fourier_features(t, period, order)
        if feats.shape[1] > 0:
            feature_blocks.append(feats)

    if not feature_blocks:
        return None
    design = np.concatenate(feature_blocks, axis=1).astype(float)  # (n_obs, K)
    n_feat = design.shape[1]
    # Honor a spec-set seasonality prior scale (priors.seasonality.prior_sigma →
    # SeasonalityConfig.prior_sigma) when present; else the passed default.
    sigma = getattr(seasonality_config, "prior_sigma", None)
    sigma = float(sigma) if sigma else float(prior_sigma)
    coefs = pm.Normal(f"{prefix}seasonality_coefs", mu=0.0, sigma=sigma, shape=n_feat)
    return pt.dot(pt.as_tensor_variable(design), coefs)


__all__ = ["build_trend_contribution", "build_seasonality_contribution"]
