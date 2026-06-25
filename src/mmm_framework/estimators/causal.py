"""Causal effect estimators beyond the back-door additive model: 2SLS (IV) and
the linear front-door estimator.

The DAG layer reports when an effect is front-door / IV *identifiable*, but until
now the framework only ever delivered the back-door additive estimate — so a
'valid IV' verdict came with a caveat that no IV estimate was actually produced.
These are real estimators (linear/Gaussian, the MMM-relevant case), so an
identifiable effect can now be *estimated* via that route.

Pure NumPy (OLS + 2SLS projection algebra); no new dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_Z975 = 1.959963984540054  # standard-normal 97.5th percentile (95% CI)


def _as_2d(a, n: int) -> np.ndarray:
    if a is None:
        return np.empty((n, 0))
    arr = np.asarray(a, dtype=float)
    return arr.reshape(n, -1)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS: returns (beta, cov_beta, residuals) with a homoskedastic covariance."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * XtX_inv
    return beta, cov, resid


@dataclass
class IVResult:
    effect: float
    se: float
    ci_low: float
    ci_high: float
    first_stage_f: float  # weak-instrument diagnostic (rule of thumb: >10 is OK)
    n: int

    @property
    def weak_instrument(self) -> bool:
        return self.first_stage_f < 10.0


def two_stage_least_squares(
    y, treatment, instruments, controls=None, ci_prob: float = 0.95
) -> IVResult:
    """2SLS estimate of a single ``treatment``'s effect on ``y`` using
    ``instruments`` (and optional ``controls``).

    The IV route survives unobserved demand confounding (the dominant MMM
    confounder) that no back-door adjustment can remove — provided a valid
    instrument exists. Linear/homoskedastic; reports a first-stage F so weak
    instruments are visible.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    T = _as_2d(treatment, n)
    if T.shape[1] != 1:
        raise ValueError("two_stage_least_squares supports a single treatment column")
    Z = _as_2d(instruments, n)
    if Z.shape[1] == 0:
        raise ValueError("at least one instrument is required")
    X = _as_2d(controls, n)

    exog = np.column_stack([np.ones(n), X])  # intercept + controls
    W = np.column_stack([T, exog])  # structural regressors (endog first)
    Zf = np.column_stack([Z, exog])  # instruments + exog

    # First stage: project the endogenous treatment onto the instrument space.
    pz = Zf @ (np.linalg.pinv(Zf.T @ Zf) @ Zf.T)
    t_hat = pz @ T[:, 0]
    W_hat = np.column_stack([t_hat, exog])  # fitted endog + exog

    WtW_inv = np.linalg.pinv(W_hat.T @ W_hat)
    beta = WtW_inv @ (W_hat.T @ y)
    # 2SLS residuals use the ORIGINAL (not fitted) regressors.
    resid = y - W @ beta
    dof = max(n - W.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * WtW_inv
    effect = float(beta[0])
    se = float(math.sqrt(max(cov[0, 0], 0.0)))

    # First-stage partial F for the excluded instruments.
    _, _, r_resid = _ols(T[:, 0], exog)  # restricted (exog only)
    _, _, f_resid = _ols(T[:, 0], Zf)  # full (exog + instruments)
    rss_r = float(r_resid @ r_resid)
    rss_f = float(f_resid @ f_resid)
    q = Z.shape[1]
    f_dof = max(n - Zf.shape[1], 1)
    first_stage_f = (
        ((rss_r - rss_f) / q) / (rss_f / f_dof) if rss_f > 1e-12 and q > 0 else math.inf
    )

    z = _Z975 if abs(ci_prob - 0.95) < 1e-9 else _norm_q((1 + ci_prob) / 2)
    return IVResult(
        effect=effect,
        se=se,
        ci_low=effect - z * se,
        ci_high=effect + z * se,
        first_stage_f=float(first_stage_f),
        n=n,
    )


@dataclass
class FrontDoorResult:
    effect: float
    se: float
    ci_low: float
    ci_high: float
    stage1_t_on_m: float  # T -> M
    stage2_m_on_y: float  # M -> Y | T
    n: int


def frontdoor_estimate(
    y, treatment, mediator, controls=None, ci_prob: float = 0.95
) -> FrontDoorResult:
    """Linear front-door estimate of ``treatment`` -> ``y`` THROUGH ``mediator``.

    Identifies the effect via the mediation pathway even when treatment and
    outcome share an *unobserved* confounder (so back-door fails): effect =
    (T->M) x (M->Y | T). Delta-method SE. Single treatment + single mediator.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    T = _as_2d(treatment, n)
    M = _as_2d(mediator, n)
    if T.shape[1] != 1 or M.shape[1] != 1:
        raise ValueError("frontdoor_estimate supports a single treatment + mediator")
    X = _as_2d(controls, n)
    exog = np.column_stack([np.ones(n), X])

    # Stage 1: M ~ T (+ controls). a = coef on T.
    s1_X = np.column_stack([T[:, 0], exog])
    b1, cov1, _ = _ols(M[:, 0], s1_X)
    a, var_a = float(b1[0]), float(cov1[0, 0])

    # Stage 2: Y ~ M + T (+ controls). b = coef on M (controlling for T).
    s2_X = np.column_stack([M[:, 0], T[:, 0], exog])
    b2, cov2, _ = _ols(y, s2_X)
    b, var_b = float(b2[0]), float(cov2[0, 0])

    effect = a * b
    # Delta method: Var(a*b) ≈ b^2 Var(a) + a^2 Var(b).
    var = b * b * var_a + a * a * var_b
    se = math.sqrt(max(var, 0.0))
    z = _Z975 if abs(ci_prob - 0.95) < 1e-9 else _norm_q((1 + ci_prob) / 2)
    return FrontDoorResult(
        effect=effect,
        se=se,
        ci_low=effect - z * se,
        ci_high=effect + z * se,
        stage1_t_on_m=a,
        stage2_m_on_y=b,
        n=n,
    )


def _norm_q(p: float) -> float:
    """Standard-normal quantile via the inverse error function (no scipy needed)."""
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    # Winitzki approximation — adequate for CI multipliers.
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)
