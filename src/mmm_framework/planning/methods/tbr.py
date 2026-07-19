"""Time-based regression (TBR) — a counterfactual for the treated time series
regressed on control time series over the pre-period, projected through the test
window; the causal effect is the cumulative ``sum_test (observed - predicted)``.

Two-tier engine (the load-bearing design nuance):

* **Fast path** (:func:`tbr_estimator`, :func:`tbr_counterfactual`) — a conjugate
  Gaussian regression predictive, numpy only. This is the registered estimator
  used inside the A/A·A/B sliding-window loop, where a full MCMC fit per window
  would be intractable. It returns a genuine predictive SD for the cumulative
  effect (parameter + observation uncertainty), which the A/A path then
  calibrates against real autocorrelation.

* **Headline path** (:func:`tbr_causal_impact`) — a full Bayesian structural
  time-series (BSTS, CausalImpact-style): a local-level state (optionally a local
  linear trend and weekly seasonality) plus the control regression, fit with
  PyMC. Used for the single experiment read-out shown in the DesignStudio /
  report. PyMC is imported **lazily** so ``import planning.methods`` stays
  PyMC-free.

References: Kerman et al. (2017) TBR; Brodersen et al. (2015) CausalImpact.
"""

from __future__ import annotations

import numpy as np

from ..simulation import Assignment, EstimatorResult, SimPanel, Window

_EPS = 1e-9


def tbr_counterfactual(
    x_pre: np.ndarray, y_pre: np.ndarray, x_test: np.ndarray
) -> dict:
    """Conjugate Gaussian regression counterfactual (fast path).

    Fits ``y_pre ~ [1, x_pre]`` and projects the counterfactual over the test
    window. Returns per-week predictive mean/sd plus the cumulative-effect
    predictive mean/sd (exact for the Gaussian model, including parameter
    covariance across the summed weeks).

    Parameters
    ----------
    x_pre, x_test : ``(t,)`` or ``(t, k)`` control regressors (pre / test).
    y_pre : ``(t_pre,)`` treated series over the pre-period.
    """
    x_pre = np.atleast_2d(np.asarray(x_pre, dtype=float))
    x_test = np.atleast_2d(np.asarray(x_test, dtype=float))
    # atleast_2d makes a 1-D series a (1, t) row — transpose to (t, 1).
    if x_pre.shape[0] == 1 and x_pre.shape[1] == len(y_pre):
        x_pre = x_pre.T
    if x_test.shape[0] == 1:
        x_test = x_test.T
    y = np.asarray(y_pre, dtype=float)
    n_pre = len(y)

    xp = np.column_stack([np.ones(n_pre), x_pre])  # (n_pre, p)
    p = xp.shape[1]
    xtx = xp.T @ xp
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ xp.T @ y
    resid = y - xp @ beta
    dof = max(n_pre - p, 1)
    s2 = float(resid @ resid) / dof

    xt = np.column_stack([np.ones(x_test.shape[0]), x_test])  # (t_test, p)
    pred_mean = xt @ beta
    # per-week predictive variance: s2 (1 + x Cov x')
    param_var = np.einsum("ij,jk,ik->i", xt, xtx_inv, xt)
    pred_var = s2 * (1.0 + param_var)
    pred_sd = np.sqrt(np.clip(pred_var, 0.0, None))

    # cumulative predictive variance: g Cov g' * s2 + s2 * t_test, g = sum_rows(xt)
    g = xt.sum(axis=0)
    cum_param_var = float(g @ xtx_inv @ g) * s2
    cum_obs_var = s2 * xt.shape[0]
    cum_sd = float(np.sqrt(max(cum_param_var + cum_obs_var, 0.0)))

    return {
        "pred_mean": pred_mean,
        "pred_sd": pred_sd,
        "cum_mean": float(pred_mean.sum()),
        "cum_sd": cum_sd,
        "sigma": float(np.sqrt(s2)),
        "beta": beta,
    }


def tbr_estimator(
    panel: SimPanel, assignment: Assignment, window: Window
) -> EstimatorResult:
    """Fast TBR point effect + predictive SE (the registered estimator).

    Uses the aggregate control series as the counterfactual regressor (classic
    TBR aggregates the control markets). Total incremental KPI =
    ``sum_test (observed_treated - predicted_counterfactual)``.
    """
    treated = list(assignment.treatment_geos)
    donors = list(assignment.control_geos)
    if not treated or not donors:
        return EstimatorResult(0.0, None, 0.0, 0)
    kpi = panel.kpi_wide
    y = kpi[treated].sum(axis=1).to_numpy(float)
    x = kpi[donors].sum(axis=1).to_numpy(float)  # aggregate control
    y_pre, x_pre = y[window.pre_slice], x[window.pre_slice]
    y_test, x_test = y[window.test_slice], x[window.test_slice]
    if len(y_pre) < 4:
        return EstimatorResult(0.0, None, 0.0, window.t_test)
    cf = tbr_counterfactual(x_pre, y_pre, x_test)
    total = float(np.sum(y_test - cf["pred_mean"]))
    return EstimatorResult(total, float(cf["cum_sd"]), 0.0, window.t_test)


def tbr_causal_impact(
    x_pre: np.ndarray,
    y_pre: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray | None = None,
    *,
    method: str = "nuts",
    trend: bool = False,
    seasonality: int | None = None,
    draws: int = 1000,
    tune: int = 1000,
    seed: int = 0,
) -> dict:
    """Full BSTS counterfactual (headline read-out; lazy PyMC import).

    A local-level state (optional local linear trend + Fourier seasonality of
    period ``seasonality``) plus a regression on the control series, fit on the
    pre-period. Forecasts the test-window counterfactual and returns the
    pointwise + cumulative causal-effect posterior.

    ``method`` is passed through to the approximate-fit engine when not ``"nuts"``
    (``"map"`` / ``"advi"`` / ``"laplace"`` for a fast read-out).
    """
    import pymc as pm  # lazy — keeps planning.methods PyMC-free

    x_pre = np.asarray(x_pre, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y = np.asarray(y_pre, dtype=float)
    n_pre = len(y)
    if x_pre.ndim == 1:
        x_pre = x_pre[:, None]
    if x_test.ndim == 1:
        x_test = x_test[:, None]
    k = x_pre.shape[1]
    t_test = x_test.shape[0]

    # standardize control regressors on the pre-period (stabilizes the level state)
    mu_x = x_pre.mean(axis=0)
    sd_x = x_pre.std(axis=0)
    sd_x[sd_x < _EPS] = 1.0
    xp = (x_pre - mu_x) / sd_x
    xt = (x_test - mu_x) / sd_x
    y_mean = float(y.mean())

    with pm.Model() as model:
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=float(np.std(y)) or 1.0)
        sigma_level = pm.HalfNormal(
            "sigma_level", sigma=(float(np.std(np.diff(y))) or 1.0)
        )
        beta = pm.Normal("beta", 0.0, sigma=2.0 * float(np.std(y) or 1.0), shape=k)
        # local level over the FULL span (pre + test) as a Gaussian random walk
        level = pm.GaussianRandomWalk(
            "level",
            sigma=sigma_level,
            init_dist=pm.Normal.dist(y_mean, 5.0),
            shape=n_pre + t_test,
        )
        contrib_pre = pm.math.dot(xp, beta)
        mu_pre = level[:n_pre] + contrib_pre
        if trend:
            drift = pm.Normal("drift", 0.0, 1.0)
            mu_pre = mu_pre + drift * np.arange(n_pre)
        if seasonality:
            t_all = np.arange(n_pre + t_test)
            fourier = np.column_stack(
                [
                    np.sin(2 * np.pi * t_all / seasonality),
                    np.cos(2 * np.pi * t_all / seasonality),
                ]
            )
            season_coef = pm.Normal("season_coef", 0.0, 1.0, shape=2)
            season = pm.math.dot(fourier, season_coef)
            mu_pre = mu_pre + season[:n_pre]
        pm.Normal("y_obs", mu=mu_pre, sigma=sigma_obs, observed=y)

        # counterfactual over the test window (what the treated series WOULD be)
        contrib_test = pm.math.dot(xt, beta)
        mu_test = level[n_pre:] + contrib_test
        if trend:
            mu_test = mu_test + drift * np.arange(n_pre, n_pre + t_test)
        if seasonality:
            mu_test = mu_test + season[n_pre:]
        pm.Deterministic("counterfactual", mu_test)

    if method and method != "nuts":
        from ...config.enums import FitMethod
        from ...model.base import run_approximate_fit

        idata, _diag = run_approximate_fit(model, FitMethod(method), draws, seed)
    else:
        with model:
            idata = pm.sample(
                draws=draws, tune=tune, chains=2, random_seed=seed, progressbar=False
            )
        with model:
            idata.extend(
                pm.sample_posterior_predictive(idata, var_names=["counterfactual"])
            )

    cf = _extract_counterfactual(idata)  # (draws, t_test)
    pred_mean = cf.mean(axis=0)
    lo, hi = np.percentile(cf, [2.5, 97.5], axis=0)

    result = {
        "pred_mean": pred_mean,
        "pred_lower": lo,
        "pred_upper": hi,
        "method": method,
    }
    if y_test is not None:
        y_test = np.asarray(y_test, dtype=float)
        pointwise = y_test[None, :] - cf  # (draws, t_test)
        cumulative = pointwise.sum(axis=1)  # (draws,)
        result.update(
            {
                "pointwise_effect": pointwise.mean(axis=0),
                "cumulative_effect": float(cumulative.mean()),
                "cumulative_lower": float(np.percentile(cumulative, 2.5)),
                "cumulative_upper": float(np.percentile(cumulative, 97.5)),
                "prob_positive": float(np.mean(cumulative > 0)),
            }
        )
    return result


def _extract_counterfactual(idata) -> np.ndarray:
    """Pull the ``counterfactual`` deterministic as a ``(draws, t_test)`` array
    from either the posterior or posterior_predictive group."""
    for group in ("posterior_predictive", "posterior"):
        if hasattr(idata, group):
            ds = getattr(idata, group)
            if "counterfactual" in getattr(ds, "data_vars", {}):
                arr = ds["counterfactual"].to_numpy()
                return arr.reshape(-1, arr.shape[-1])
    raise KeyError("counterfactual not found in idata")


__all__ = ["tbr_counterfactual", "tbr_estimator", "tbr_causal_impact"]
