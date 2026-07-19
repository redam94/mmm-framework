"""Synthetic control (Abadie-Diamond-Hainmueller) — a donor-pool counterfactual
with **convex** weights and **placebo-permutation** inference.

This is the proper SCM the loose ``regadj_geo_estimator`` in
:mod:`planning.simulation` only approximates: the donor weights are constrained
to the simplex (``w >= 0``, ``sum w = 1``) so the synthetic control cannot
extrapolate outside the convex hull of the donor pool, and inference comes from
re-running the whole procedure with each donor as a *placebo* treated unit — the
distribution of placebo effects is the null the actual effect is judged against.

The registered estimator (``synthetic_control_estimator``) is the fast path used
inside the A/A·A/B simulation loop: it returns the point effect with ``se=None``
so the harness builds the null empirically across windows (running the O(n_donor)
placebo permutation inside every sliding window would be needlessly expensive).
The single headline read-out uses :func:`synthetic_control_analysis`, which adds
the placebo distribution, a permutation p-value, and the standard RMSPE
robustness screen.

numpy + scipy only (kernel-safe).
"""

from __future__ import annotations

import math

import numpy as np

from ..simulation import Assignment, EstimatorResult, SimPanel, Window

_EPS = 1e-9


def synthetic_control_weights(
    x_pre_donors: np.ndarray, y_pre_treated: np.ndarray
) -> np.ndarray:
    """Convex donor weights minimizing pre-period fit ``||X w - y||^2`` subject to
    ``w >= 0`` and ``sum w = 1`` (the ADH simplex constraint).

    Parameters
    ----------
    x_pre_donors : ``(t_pre, n_donors)`` donor KPI over the pre-period.
    y_pre_treated : ``(t_pre,)`` treated KPI over the pre-period.
    """
    x = np.asarray(x_pre_donors, dtype=float)
    y = np.asarray(y_pre_treated, dtype=float)
    n = x.shape[1]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    w0 = np.full(n, 1.0 / n)

    def _loss(w: np.ndarray) -> float:
        r = x @ w - y
        return float(r @ r)

    def _grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * x.T @ (x @ w - y)

    try:
        from scipy.optimize import minimize

        res = minimize(
            _loss,
            w0,
            jac=_grad,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
            options={"maxiter": 300, "ftol": 1e-10},
        )
        w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
    except Exception:
        # Fallback: non-negative least squares, then renormalize onto the simplex.
        from numpy.linalg import lstsq

        w, *_ = lstsq(x, y, rcond=None)
        w = np.clip(w, 0.0, None)

    s = float(w.sum())
    return w / s if s > _EPS else np.full(n, 1.0 / n)


def _treated_donor_series(
    panel: SimPanel, treated: list[str], donors: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    kpi = panel.kpi_wide
    y = kpi[treated].sum(axis=1).to_numpy(float)  # aggregate treated series
    x = kpi[donors].to_numpy(float)  # weeks x n_donors
    return y, x


def _scm_effect(
    y: np.ndarray, x: np.ndarray, window: Window
) -> tuple[float, float, float]:
    """Return ``(total_effect, pre_rmspe, post_rmspe)`` for one treated/donor split.

    Uses **demeaned** SCM: convex donor weights are fit to pre-period *deviations*
    and the treated level is carried separately. This is what lets a treated
    aggregate (multiple treated geos, so a higher level than any single donor) be
    reconstructed by a convex combination of individual donors — without demeaning,
    ``sum(w)=1`` pins the synthetic control to a single-geo level and the level
    mismatch masquerades as a huge effect.
    """
    x_pre, y_pre = x[window.pre_slice], y[window.pre_slice]
    x_test, y_test = x[window.test_slice], y[window.test_slice]
    mu_y = float(y_pre.mean())
    mu_x = x_pre.mean(axis=0)
    w = synthetic_control_weights(x_pre - mu_x, y_pre - mu_y)
    pred_pre = mu_y + (x_pre - mu_x) @ w
    pred_test = mu_y + (x_test - mu_x) @ w
    total = float(np.sum(y_test - pred_test))
    pre_rmspe = (
        float(np.sqrt(np.mean((y_pre - pred_pre) ** 2))) if len(y_pre) else float("nan")
    )
    post_gap = y_test - pred_test
    post_rmspe = float(np.sqrt(np.mean(post_gap**2))) if len(post_gap) else float("nan")
    return total, pre_rmspe, post_rmspe


def synthetic_control_estimator(
    panel: SimPanel, assignment: Assignment, window: Window
) -> EstimatorResult:
    """Fast SCM point effect (``se=None`` → the A/A path calibrates the null).

    Total incremental KPI = ``sum_test (observed_treated - synthetic_control)``.
    """
    treated = list(assignment.treatment_geos)
    donors = list(assignment.control_geos)
    if not treated or not donors:
        return EstimatorResult(0.0, None, 0.0, 0)
    y, x = _treated_donor_series(panel, treated, donors)
    total, _pre, _post = _scm_effect(y, x, window)
    return EstimatorResult(float(total), None, 0.0, window.t_test)


def placebo_distribution(
    panel: SimPanel, assignment: Assignment, window: Window
) -> np.ndarray:
    """Re-run SCM treating each *donor* as a fake treated unit (its donor pool =
    the remaining donors). The distribution of placebo total-effects is the null."""
    donors = list(assignment.control_geos)
    placebos: list[float] = []
    for i, g in enumerate(donors):
        pool = donors[:i] + donors[i + 1 :]
        if not pool:
            continue
        y, x = _treated_donor_series(panel, [g], pool)
        total, _pre, _post = _scm_effect(y, x, window)
        if math.isfinite(total):
            placebos.append(total)
    return np.asarray(placebos, dtype=float)


def synthetic_control_analysis(
    panel: SimPanel,
    assignment: Assignment,
    window: Window,
    *,
    rmspe_screen: float | None = 5.0,
) -> dict:
    """Headline SCM read-out: point effect + placebo-permutation inference.

    ``rmspe_screen`` drops placebo units whose pre-period fit is more than this
    multiple of the treated unit's pre-RMSPE (the ADH robustness screen — a
    placebo that never fit its own pre-period would inflate the null). ``None``
    disables the screen.
    """
    treated = list(assignment.treatment_geos)
    donors = list(assignment.control_geos)
    y, x = _treated_donor_series(panel, treated, donors)
    total, pre_rmspe, post_rmspe = _scm_effect(y, x, window)

    placebos: list[tuple[float, float]] = []
    for i, g in enumerate(donors):
        pool = donors[:i] + donors[i + 1 :]
        if not pool:
            continue
        yp, xp = _treated_donor_series(panel, [g], pool)
        p_total, p_pre, _p_post = _scm_effect(yp, xp, window)
        if not math.isfinite(p_total):
            continue
        if (
            rmspe_screen is not None
            and math.isfinite(pre_rmspe)
            and math.isfinite(p_pre)
            and p_pre > rmspe_screen * max(pre_rmspe, _EPS)
        ):
            continue
        placebos.append((p_total, p_pre))

    # Rank by the RMSPE-STANDARDIZED effect (Abadie 2010/2015): |effect| / pre_rmspe.
    # A raw-effect ranking is invalid when the treated unit (here an aggregate of
    # several geos) has a different scale than single-geo placebos — standardizing
    # by each unit's own pre-period fit quality makes them comparable.
    placebo_raw = np.array([p[0] for p in placebos], dtype=float)
    placebo_std = np.array([abs(p[0]) / max(p[1], _EPS) for p in placebos], dtype=float)
    treated_std = abs(total) / max(pre_rmspe, _EPS)
    if placebo_std.size:
        p_value = float(
            (1 + np.sum(placebo_std >= treated_std)) / (1 + placebo_std.size)
        )
        null_sd = float(np.std(placebo_raw)) if placebo_raw.size > 1 else float("nan")
    else:
        p_value = float("nan")
        null_sd = float("nan")

    return {
        "estimate": float(total),
        "standardized_effect": float(treated_std),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "placebo_p_value": p_value,
        "placebo_null_sd": null_sd,
        "n_placebos": int(placebo_std.size),
        "null_method": "placebo_permutation",
    }


__all__ = [
    "synthetic_control_weights",
    "synthetic_control_estimator",
    "synthetic_control_analysis",
    "placebo_distribution",
]
