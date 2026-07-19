"""Geo-based regression (GBR) — Google's cross-sectional geo experiment
estimator: one row per geo, regress the test-period response on the pre-period
response and the treatment assignment; the incremental effect is the coefficient
on the treatment term.

Classic GBR regresses ``y_post_g`` on ``y_pre_g`` and the per-geo spend change,
so the treatment coefficient is an incremental-response-per-dollar (iROAS). Inside
the A/A·A/B simulation harness there is no real spend change (the lift is injected
directly onto the treated geos' KPI), so the estimator uses a **treatment
indicator** instead — the coefficient is then the per-geo incremental KPI, summed
over treated geos to a total. When per-geo spend deltas are supplied (the real
read-out), :func:`gbr_iroas` divides the total effect by the treated spend change.

numpy only (kernel-safe).
"""

from __future__ import annotations

import math

import numpy as np

from ..simulation import Assignment, EstimatorResult, SimPanel, Window


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares → (coef, cov). ``w`` are observation weights."""
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    xtx = Xw.T @ Xw
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    coef = xtx_inv @ Xw.T @ yw
    resid = yw - Xw @ coef
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * xtx_inv
    return coef, cov


def gbr_estimator(
    panel: SimPanel,
    assignment: Assignment,
    window: Window,
    *,
    weighted: bool = True,
) -> EstimatorResult:
    """Cross-sectional GBR: ``y_post_g ~ 1 + y_pre_g + treat_g``.

    Total incremental KPI = ``theta_treat * n_treated`` (per-geo incremental
    effect times the number of treated geos). The SE on ``theta`` is scaled to
    the total.
    """
    treated = set(assignment.treatment_geos)
    donors = list(assignment.control_geos)
    geos = list(assignment.treatment_geos) + donors
    geos = list(dict.fromkeys(geos))  # dedupe, keep order
    if len(geos) < 3:
        return EstimatorResult(0.0, None, 0.0, len(geos))
    kpi = panel.kpi_wide

    y_post = np.array([kpi[g].to_numpy(float)[window.test_slice].sum() for g in geos])
    y_pre = np.array([kpi[g].to_numpy(float)[window.pre_slice].sum() for g in geos])
    treat = np.array([1.0 if g in treated else 0.0 for g in geos])

    X = np.column_stack([np.ones(len(geos)), y_pre, treat])
    # weight geos by pre-period size (heteroskedasticity: bigger geos, bigger noise)
    if weighted:
        w = np.clip(y_pre, 1.0, None)
        w = w / w.mean()
    else:
        w = np.ones(len(geos))

    coef, cov = _wls(y_post, X, w)
    theta = float(coef[2])  # treatment coefficient = per-geo incremental KPI
    se_theta = float(math.sqrt(max(cov[2, 2], 0.0)))
    n_treated = int(treat.sum())
    total = theta * n_treated
    se_total = se_theta * n_treated
    return EstimatorResult(float(total), float(se_total), 0.0, len(geos))


def gbr_iroas(total_effect: float, treated_spend_delta: float) -> float | None:
    """Incremental ROAS from a GBR total effect and the treated spend change."""
    if not treated_spend_delta or not math.isfinite(treated_spend_delta):
        return None
    return float(total_effect / treated_spend_delta)


__all__ = ["gbr_estimator", "gbr_iroas"]
