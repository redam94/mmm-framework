"""Facts layer for the interactive MMM Results Report.

Everything the interactive report shows is computed here, once, from a fitted
:class:`~mmm_framework.model.base.BayesianMMM` — per-draw, per-period channel
contributions in KPI units, posterior-predictive fit bands, response-curve
grids, carryover kernels, prior-vs-posterior estimand densities, and a
no-refit sensitivity battery. The client-side JavaScript only ever
*re-aggregates* these draws over user-selected windows; it never invents new
numbers, so every interactive readout is a genuine posterior summary.

Per-draw matrices are embedded as base64-encoded little-endian ``float32``
buffers (``draws`` × ``periods`` row-major) to keep the standalone HTML small.

The window-recomputation contract (shared with the report's JS):

- ``contribution ROI`` over a window = ``sum_t contrib[d, t] / sum_t spend[t]``
  per draw ``d`` — same semantics as the ``contribution_roi`` estimand.
- ``marginal ROAS`` over a window = ``sum_t dcontrib[d, t] / sum_t dspend[t]``
  where ``dcontrib`` is the paired contribution delta under an all-channel
  ``+bump_pct%`` spend perturbation (posterior params held fixed per draw).
- Intervals are central (equal-tailed) credible intervals, matching
  :func:`mmm_framework.utils.compute_hdi_bounds`.
"""

from __future__ import annotations

import base64
import re
import warnings
from typing import Any

import numpy as np
from loguru import logger

from ...utils import compute_hdi_bounds
from ..helpers.measurement import resolve_channel_divisor
from ..helpers.prefit import (
    _adstock_weights,
    model_assumptions,
    prior_estimand_facts,
    prior_predictive_facts,
    sample_prior,
)

__all__ = ["interactive_report_facts"]

#: Nested predictive-band levels rendered as the fit-plot gradient.
FIT_BAND_LEVELS = (0.5, 0.8, 0.9, 0.95)

#: Default spend-multiplier grid for response / ROI / mROI curves and the
#: budget reallocator (0 → 2× current spend).
DEFAULT_CURVE_MULTIPLIERS = tuple(np.round(np.linspace(0.0, 2.0, 21), 3))


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────
def _b64_f32(arr: np.ndarray) -> str:
    """Base64-encode an array as little-endian float32 (row-major)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def _jsafe(x: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-safe python values."""
    if isinstance(x, dict):
        return {str(k): _jsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsafe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsafe(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _flat_posterior(trace: Any, name: str) -> np.ndarray | None:
    """Flatten a posterior variable to ``(n_samples, *shape)``; None if absent."""
    post = getattr(trace, "posterior", None)
    if post is None or name not in getattr(post, "data_vars", {}):
        return None
    vals = np.asarray(post[name].values, dtype=float)
    return vals.reshape(-1, *vals.shape[2:])


def _period_onehot(time_idx: np.ndarray, n_periods: int) -> np.ndarray:
    """One-hot scatter matrix ``(n_obs, n_periods)``: sums obs into periods."""
    m = np.zeros((time_idx.shape[0], n_periods))
    m[np.arange(time_idx.shape[0]), time_idx] = 1.0
    return m


def _resolve_periods(model: Any) -> Any:
    """Period index for the x-axis, tolerant of the model's data layout.

    Core :class:`BayesianMMM` carries a ``panel`` with ``coords.periods``; the
    extension models (:class:`BaseExtendedMMM` family) are single national
    series that only carry a time ``index``. Fall back through both, then to a
    plain integer range keyed off the outcome length so any fitted model with a
    ``channel_contributions`` surface can be reported.
    """
    coords = getattr(getattr(model, "panel", None), "coords", None)
    periods = getattr(coords, "periods", None)
    if periods is not None:
        return periods
    index = getattr(model, "index", None)
    if index is not None:
        return index
    y = getattr(model, "y_raw", None)
    if y is None:
        y = getattr(model, "y", [])
    return np.arange(len(np.asarray(y)))


def _eti(draws: np.ndarray, interval: float) -> tuple[float, float]:
    lo_q = (1.0 - interval) / 2.0 * 100.0
    return (
        float(np.percentile(draws, lo_q)),
        float(np.percentile(draws, 100.0 - lo_q)),
    )


def _series_stats(
    actual: np.ndarray,
    pred_mean: np.ndarray,
    band90: tuple[np.ndarray, np.ndarray],
) -> dict[str, float | None]:
    """Fit statistics for one observed-vs-predicted period series."""
    ok = np.isfinite(actual) & np.isfinite(pred_mean)
    if ok.sum() < 3:
        return {"r2": None, "rmse": None, "mape": None, "coverage90": None}
    a, p = actual[ok], pred_mean[ok]
    resid = a - p
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else None
    rmse = float(np.sqrt(np.mean(resid**2)))
    nz = np.abs(a) > 1e-9
    mape = float(np.mean(np.abs(resid[nz] / a[nz])) * 100.0) if nz.sum() >= 3 else None
    lo, hi = band90
    okc = ok & np.isfinite(lo) & np.isfinite(hi)
    coverage = (
        float(np.mean((actual[okc] >= lo[okc]) & (actual[okc] <= hi[okc])))
        if okc.sum() >= 3
        else None
    )
    return {"r2": r2, "rmse": rmse, "mape": mape, "coverage90": coverage}


# ─────────────────────────────────────────────────────────────────────────────
# Fit-plot facts (posterior predictive per geo + national)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_series(
    samples: np.ndarray,
    actual_obs: np.ndarray,
    time_idx: np.ndarray,
    n_periods: int,
    obs_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Aggregate obs-level predictive draws + actuals to one period series.

    Rows (geo × product) inside the selection are *summed* per period. Periods
    with no observations become ``None`` in the output.
    """
    if obs_mask is not None:
        if not obs_mask.any():
            return None
        samples = samples[:, obs_mask]
        actual_obs = actual_obs[obs_mask]
        time_idx = time_idx[obs_mask]
    onehot = _period_onehot(time_idx, n_periods)
    counts = onehot.sum(axis=0)
    covered = counts > 0
    if covered.sum() < 3:
        return None

    agg = samples @ onehot  # (n_draws, n_periods)
    actual = actual_obs @ onehot
    actual[~covered] = np.nan
    agg[:, ~covered] = np.nan

    bands: dict[str, dict[str, list]] = {}
    band90: tuple[np.ndarray, np.ndarray] | None = None
    for prob in FIT_BAND_LEVELS:
        lo, hi = compute_hdi_bounds(agg, hdi_prob=prob, axis=0)
        key = str(int(round(prob * 100)))
        bands[key] = {"lo": _jsafe(lo), "hi": _jsafe(hi)}
        if key == "90":
            band90 = (lo, hi)

    mean = np.nanmean(agg, axis=0)
    stats = _series_stats(actual, mean, band90 if band90 else (mean, mean))
    return {
        "actual": _jsafe(actual),
        "mean": _jsafe(mean),
        "bands": bands,
        "stats": stats,
    }


def _fit_facts(
    model: Any,
    time_idx: np.ndarray,
    n_periods: int,
    random_seed: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit-plot facts per series + posterior-predictive test-statistic facts
    (both derive from the same, single ``predict()`` pass)."""
    pred = model.predict(return_original_scale=True, random_seed=random_seed)
    samples = np.asarray(pred.y_pred_samples, dtype=float)
    actual_obs = np.asarray(model.y_raw, dtype=float)

    series: dict[str, Any] = {}
    order: list[str] = []
    national = _fit_series(samples, actual_obs, time_idx, n_periods)
    if national is not None:
        series["National"] = national
        order.append("National")

    geo_names = list(getattr(model, "geo_names", []) or [])
    geo_idx = np.asarray(getattr(model, "geo_idx", np.zeros(0)), dtype=int)
    if len(geo_names) > 1 and geo_idx.shape[0] == samples.shape[1]:
        for g, name in enumerate(geo_names):
            entry = _fit_series(
                samples, actual_obs, time_idx, n_periods, obs_mask=geo_idx == g
            )
            if entry is not None:
                series[str(name)] = entry
                order.append(str(name))
    fit = {"series": series, "order": order, "band_levels": list(FIT_BAND_LEVELS)}

    onehot = _period_onehot(time_idx, n_periods)
    covered = onehot.sum(axis=0) > 0
    rep = (samples @ onehot)[:, covered]
    obs = (actual_obs @ onehot)[covered]
    ppc_stats = _ppc_stat_facts(rep, obs)
    try:
        pit = _loo_pit_facts(model, samples, actual_obs)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"interactive report: LOO-PIT facts failed: {e}")
        pit = None
    if pit is not None:
        ppc_stats["loo_pit"] = pit
    return fit, ppc_stats


# ─────────────────────────────────────────────────────────────────────────────
# Posterior-predictive test statistics (Bayesian p-values)
# ─────────────────────────────────────────────────────────────────────────────
#: Test statistics compared between replicated and observed national KPI
#: series. Each is T(y): the Bayesian p-value is P(T(y_rep) >= T(y_obs)).
PPC_TEST_STATS = (
    ("mean", "Mean", "average level"),
    ("sd", "Std. deviation", "period-to-period variability"),
    ("min", "Minimum", "the worst period"),
    ("max", "Maximum", "the best period"),
    ("acf1", "Lag-1 autocorrelation", "week-to-week persistence (AR(1))"),
    ("skew", "Skewness", "asymmetry of the KPI distribution"),
)


def _stat_over_rows(key: str, x: np.ndarray) -> np.ndarray:
    """One test statistic per row of ``x`` (rows = replicated series).

    NaN periods are ignored; for ``acf1`` the series is treated as contiguous
    across any gap (national series are normally gap-free).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        if key == "mean":
            return np.nanmean(x, axis=1)
        if key == "sd":
            return np.nanstd(x, axis=1)
        if key == "min":
            return np.nanmin(x, axis=1)
        if key == "max":
            return np.nanmax(x, axis=1)
        if key == "skew":
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            sd = np.where(sd < 1e-12, np.nan, sd)
            return np.nanmean(((x - mu) / sd) ** 3, axis=1)
        if key == "acf1":
            a, b = x[:, :-1], x[:, 1:]
            ok = np.isfinite(a) & np.isfinite(b)
            a = np.where(ok, a, np.nan)
            b = np.where(ok, b, np.nan)
            ma = np.nanmean(a, axis=1, keepdims=True)
            mb = np.nanmean(b, axis=1, keepdims=True)
            cov = np.nanmean((a - ma) * (b - mb), axis=1)
            sa = np.sqrt(np.nanmean((a - ma) ** 2, axis=1))
            sb = np.sqrt(np.nanmean((b - mb) ** 2, axis=1))
            den = sa * sb
            return np.where(den < 1e-12, np.nan, cov / den)
    return np.full(x.shape[0], np.nan)


def _ppc_stat_facts(
    rep: np.ndarray, obs: np.ndarray, n_bins: int = 27
) -> dict[str, Any]:
    """Replicated-vs-observed test statistics with Bayesian p-values.

    ``rep``: ``(n_draws, P)`` replicated national KPI series; ``obs``: ``(P,)``
    observed. A p-value near 0 or 1 means the model systematically fails to
    reproduce that property of the data.
    """
    rows: list[dict[str, Any]] = []
    if rep.ndim != 2 or rep.shape[0] < 20 or obs.size < 4:
        return {"stats": rows, "n_draws": int(rep.shape[0]) if rep.ndim == 2 else 0}
    for key, label, desc in PPC_TEST_STATS:
        obs_val = _stat_over_rows(key, obs[None, :])[0]
        if not np.isfinite(obs_val):
            continue
        vals = _stat_over_rows(key, rep)
        vals = vals[np.isfinite(vals)]
        if vals.size < 20:
            continue
        p = float(np.mean(vals >= obs_val))
        lo = float(min(vals.min(), obs_val))
        hi = float(max(vals.max(), obs_val))
        pad = 0.02 * (hi - lo) if hi > lo else max(abs(obs_val), 1.0) * 0.05
        edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
        counts, _ = np.histogram(vals, bins=edges)
        rows.append(
            {
                "key": key,
                "label": label,
                "desc": desc,
                "observed": float(obs_val),
                "rep_mean": float(vals.mean()),
                "bayes_p": p,
                "extreme": bool(p < 0.05 or p > 0.95),
                "hist": {
                    "edges": _jsafe(edges),
                    "counts": [int(c) for c in counts],
                },
            }
        )
    return {
        "stats": rows,
        "n_draws": int(rep.shape[0]),
        "series": "national period-summed KPI",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOO-PIT predictive calibration
# ─────────────────────────────────────────────────────────────────────────────
def _pointwise_loglik(model: Any, n_samples: int) -> np.ndarray | None:
    """Pointwise log-likelihood draws aligned with ``predict()``'s flat order.

    ``predict()`` reshapes ``(chain, draw, obs) -> (chain*draw, obs)`` in C
    order; ``pm.compute_log_likelihood`` returns the same layout over the same
    trace, so row ``s`` here pairs exactly with predictive draw ``s``. Returns
    ``(n_samples, n_obs)`` or ``None`` when the log-likelihood cannot be
    computed (duck-typed models, graph/trace mismatch, non-scalar outcome).
    """
    trace = getattr(model, "_trace", None)
    if trace is None:
        return None
    from ...utils.arviz_compat import has_group

    ll_ds = None
    try:
        if has_group(trace, "log_likelihood"):
            ll_ds = trace.log_likelihood
    except Exception:  # noqa: BLE001
        ll_ds = None
    if ll_ds is None or not len(getattr(ll_ds, "data_vars", {})):
        try:
            import pymc as pm

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ll_ds = pm.compute_log_likelihood(
                    trace,
                    model=model.model,
                    extend_inferencedata=False,
                    progressbar=False,
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"interactive report: pointwise log-lik unavailable: {e}")
            return None
    try:
        names = list(ll_ds.data_vars)
        var = "y_obs" if "y_obs" in names else names[0]
        arr = np.asarray(ll_ds[var].values, dtype=float)
        if arr.ndim != 3:  # (chain, draw, obs) only — multi-dim outcomes bail
            return None
        ll = arr.reshape(-1, arr.shape[-1])
    except Exception:  # noqa: BLE001
        return None
    if ll.shape[0] != n_samples:
        return None
    return ll


def _pit_hist_band(n: int, n_bins: int, prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Simultaneous band for PIT-histogram bin counts under Uniform(0, 1).

    Continuous PIT values make the null exactly ``Multinomial(n, 1/K)``; the
    multiplicity-adjusted per-bin level comes from the SBC band machinery.
    """
    from scipy import stats as sps

    from ...diagnostics.sbc import _simultaneous_gamma

    probs = np.full(n_bins, 1.0 / n_bins)
    g = _simultaneous_gamma(n, probs, prob)
    lower = sps.binom.ppf((1.0 - g) / 2.0, n, probs)
    upper = sps.binom.ppf((1.0 + g) / 2.0, n, probs)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _loo_pit_facts(
    model: Any,
    samples: np.ndarray,
    actual_obs: np.ndarray,
    n_bins: int = 20,
    band_prob: float = 0.95,
) -> dict[str, Any] | None:
    """LOO-PIT calibration facts from the shared ``predict()`` pass.

    ``PIT_i = Σ_s w_is · 1[y_rep_is ≤ y_i]`` with ``w`` the PSIS-smoothed
    leave-one-out importance weights; under a calibrated predictive
    distribution the PIT values are Uniform(0, 1). The weights need the
    pointwise log-likelihood — when that is unavailable the check degrades to
    ordinary posterior-predictive PIT (uniform weights) and ``weighting``
    says so. Both the histogram band and the ECDF-difference band are
    simultaneous (the whole curve stays inside ``band_prob`` of the time
    under calibration). All panels are static — computed here, per
    observation, not window-recomputed.
    """
    from ...diagnostics.sbc import ecdf_diff_band
    from ...validation.calibration import loo_pit_check

    samples = np.asarray(samples, dtype=float)
    actual_obs = np.asarray(actual_obs, dtype=float).ravel()
    n_samples = int(samples.shape[0]) if samples.ndim == 2 else 0
    if samples.ndim != 2 or n_samples < 30 or actual_obs.size < 8:
        return None

    log_weights = None
    weighting = "posterior-predictive"
    khat_info: dict[str, Any] | None = None
    ll = _pointwise_loglik(model, n_samples)
    if ll is not None:
        try:
            from ...utils.arviz_compat import psis_log_weights

            lw, khat = psis_log_weights(ll.T)  # (n_obs, S), (n_obs,)
            log_weights = lw
            weighting = "psis-loo"
            khat = np.asarray(khat, dtype=float)
            khat_info = {
                "max": float(np.nanmax(khat)),
                "n_high": int(np.sum(khat > 0.7)),
                "n": int(khat.size),
            }
        except Exception as e:  # noqa: BLE001
            logger.debug(f"interactive report: PSIS smoothing failed: {e}")

    try:
        res = loo_pit_check(y=actual_obs, y_hat=samples.T, log_weights=log_weights)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"interactive report: LOO-PIT unavailable: {e}")
        return None

    pit = np.clip(res.pit, 0.0, 1.0)
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    lo, hi = _pit_hist_band(int(pit.size), n_bins, band_prob)
    z, elo, ehi = ecdf_diff_band(int(pit.size), prob=band_prob)
    ediff = np.searchsorted(np.sort(pit), z, side="right") / pit.size - z

    return {
        "n": int(res.n),
        "n_draws": n_samples,
        "weighting": weighting,
        "ks_stat": float(res.ks_stat),
        "ks_p": float(res.ks_pvalue),
        "calibrated": bool(res.calibrated),
        "khat": khat_info,
        "hist": {"edges": _jsafe(edges), "counts": [int(c) for c in counts]},
        "band": {"lo": _jsafe(lo), "hi": _jsafe(hi), "prob": band_prob},
        "ecdf": {
            "z": _jsafe(z),
            "diff": _jsafe(ediff),
            "lo": _jsafe(elo),
            "hi": _jsafe(ehi),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spend / divisor series
# ─────────────────────────────────────────────────────────────────────────────
def _divisor_series(
    model: Any,
    channels: list[str],
    time_idx: np.ndarray,
    n_periods: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Per-period divisor (spend-equivalent) series + metric meta per channel."""
    spend: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    for ch in channels:
        series = np.zeros(n_periods)
        for t in range(n_periods):
            div = resolve_channel_divisor(model, ch, mask=time_idx == t)
            series[t] = div.total
        full = resolve_channel_divisor(model, ch)
        m = full.meta.to_dict() if hasattr(full.meta, "to_dict") else {}
        m["found"] = bool(full.found)
        m["total"] = float(full.total)
        spend[ch] = series
        meta[ch] = m
    return spend, meta


# ─────────────────────────────────────────────────────────────────────────────
# Carryover (posterior adstock kernels)
# ─────────────────────────────────────────────────────────────────────────────
def _channel_transform_config(model: Any, ch: str) -> tuple[str, str, int, bool]:
    sat_family, ad_family, l_max, normalize = "logistic", "geometric", 8, True
    mff = getattr(model, "mff_config", None)
    if mff is not None:
        try:
            cfg = mff.get_media_config(ch)
            if cfg is not None:
                sat_family = cfg.saturation.type.value
                ad_family = cfg.adstock.type.value
                l_max = int(cfg.adstock.l_max)
                normalize = bool(cfg.adstock.normalize)
        except Exception:  # noqa: BLE001
            pass
    return sat_family, ad_family, l_max, normalize


def _half_life_draws(kernels: np.ndarray) -> np.ndarray:
    """Per-draw carryover half-life: interpolated lag where cumulative weight
    reaches 50% of the kernel's total."""
    cum = np.cumsum(kernels, axis=1)
    total = cum[:, -1:] + 1e-12
    share = cum / total
    n_draws, n_lags = share.shape
    out = np.zeros(n_draws)
    for d in range(n_draws):
        idx = int(np.searchsorted(share[d], 0.5))
        if idx == 0:
            out[d] = 0.5 * (0.5 / max(share[d, 0], 1e-12))
        elif idx >= n_lags:
            out[d] = float(n_lags - 1)
        else:
            lo, hi = share[d, idx - 1], share[d, idx]
            frac = (0.5 - lo) / max(hi - lo, 1e-12)
            out[d] = (idx - 1) + frac
    return out


def _carryover_facts(
    model: Any,
    trace: Any,
    channels: list[str],
    interval: float,
    max_draws: int = 500,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ch in channels:
        _, ad_family, l_max, normalize = _channel_transform_config(model, ch)
        params: dict[str, np.ndarray] = {}
        for key, rv in (
            ("alpha", f"adstock_alpha_{ch}"),
            ("theta", f"adstock_theta_{ch}"),
            ("shape", f"adstock_shape_{ch}"),
            ("scale", f"adstock_scale_{ch}"),
        ):
            v = _flat_posterior(trace, rv)
            if v is not None:
                params[key] = v.reshape(-1)[:max_draws]
        if not params:
            continue
        n = min(v.size for v in params.values())
        kernels = _adstock_weights(
            ad_family, l_max, {k: v[:n] for k, v in params.items()}, normalize
        )
        if kernels is None or not kernels.size:
            continue
        hl = _half_life_draws(kernels)
        hl_lo, hl_hi = _eti(hl, interval)
        lo_q = (1.0 - interval) / 2.0 * 100.0
        out[ch] = {
            "family": ad_family,
            "lags": list(range(l_max + 1)),
            "median": _jsafe(np.median(kernels, axis=0)),
            "lower": _jsafe(np.percentile(kernels, lo_q, axis=0)),
            "upper": _jsafe(np.percentile(kernels, 100.0 - lo_q, axis=0)),
            "half_life": {
                "mean": float(hl.mean()),
                "lower": hl_lo,
                "upper": hl_hi,
            },
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Prior vs posterior on the estimand scale
# ─────────────────────────────────────────────────────────────────────────────
def _kde_curve(draws: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    draws = draws[np.isfinite(draws)]
    if draws.size < 10 or float(np.std(draws)) < 1e-12:
        return None
    try:
        from scipy.stats import gaussian_kde

        return gaussian_kde(draws)(grid)
    except Exception:  # noqa: BLE001
        return None


def _prior_posterior_rows(
    channels: list[str],
    prior_est: dict[str, Any],
    posterior_roi_draws: dict[str, np.ndarray],
    divisor_meta: dict[str, dict[str, Any]],
    interval: float,
) -> list[dict[str, Any]]:
    """Per-channel prior vs posterior densities of the contribution-ROI
    estimand (or the efficiency analogue) — the decision scale, not raw
    coefficients."""
    prior_by_channel = {
        str(r.get("channel")): r for r in (prior_est or {}).get("channels", [])
    }
    rows: list[dict[str, Any]] = []
    for ch in channels:
        post = posterior_roi_draws.get(ch)
        if post is None or not np.isfinite(post).any():
            continue
        post = post[np.isfinite(post)]
        prior_row = prior_by_channel.get(ch, {})
        prior_draws = np.asarray(prior_row.get("draws", []), dtype=float)
        prior_draws = prior_draws[np.isfinite(prior_draws)]

        pool = np.concatenate([post, prior_draws]) if prior_draws.size else post
        g_lo, g_hi = np.percentile(pool, [0.5, 99.5])
        if not np.isfinite(g_lo) or not np.isfinite(g_hi) or g_hi <= g_lo:
            continue
        pad = 0.05 * (g_hi - g_lo)
        grid = np.linspace(g_lo - pad, g_hi + pad, 200)

        post_lo, post_hi = _eti(post, interval)
        meta = divisor_meta.get(ch, {})
        ref = meta.get("reference", 1.0)
        row: dict[str, Any] = {
            "channel": ch,
            "label": meta.get("roi_label") or "ROI",
            "reference": ref,
            "grid": _jsafe(grid),
            "posterior": {
                "density": _jsafe(_kde_curve(post, grid)),
                "mean": float(post.mean()),
                "sd": float(post.std()),
                "lower": post_lo,
                "upper": post_hi,
                "p_above": float(np.mean(post > float(ref))),
            },
        }
        if prior_draws.size:
            pr_lo, pr_hi = _eti(prior_draws, interval)
            row["prior"] = {
                "density": _jsafe(_kde_curve(prior_draws, grid)),
                "mean": float(prior_draws.mean()),
                "sd": float(prior_draws.std()),
                "lower": pr_lo,
                "upper": pr_hi,
                "p_above": float(np.mean(prior_draws > float(ref))),
            }
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Year-over-year drivers (for the waterfall + grounded gloss)
# ─────────────────────────────────────────────────────────────────────────────
def _yoy_facts(
    actual_national: np.ndarray,
    contrib_dp: dict[str, np.ndarray],
    periods: list[str],
    channels: list[str],
    interval: float,
) -> dict[str, Any] | None:
    """Latest-pair YoY decomposition (grounds the gloss + gates the section).

    The waterfall itself recomputes any year pair client-side from the same
    embedded draws; this mirrors that math in Python for the most recent
    qualifying pair: per-channel contribution deltas (posterior draws → CI)
    plus the residual "baseline & other" delta that closes to the observed
    total change.
    """
    year_idx: dict[str, list[int]] = {}
    for i, p in enumerate(periods):
        year_idx.setdefault(p[:4], []).append(i)
    # Only (near-)complete calendar years qualify: totals from a partial year
    # are not comparable to a full one, so a 26-week year would make the
    # bridge read as a collapse that is really just missing data.
    qual = [y for y in sorted(year_idx) if len(year_idx[y]) >= 50]
    if len(qual) < 2:
        return None
    ya, yb = qual[-2], qual[-1]
    ia = np.array(year_idx[ya], dtype=int)
    ib = np.array(year_idx[yb], dtype=int)
    tot_a = float(np.nansum(actual_national[ia]))
    tot_b = float(np.nansum(actual_national[ib]))

    drivers: list[dict[str, Any]] = []
    media_delta = None
    for ch in channels:
        d = contrib_dp[ch][:, ib].sum(axis=1) - contrib_dp[ch][:, ia].sum(axis=1)
        media_delta = d if media_delta is None else media_delta + d
        lo, hi = _eti(d, interval)
        drivers.append({"name": ch, "mean": float(d.mean()), "lower": lo, "upper": hi})
    base = (tot_b - tot_a) - media_delta
    b_lo, b_hi = _eti(base, interval)
    return {
        "years": qual,
        "latest": {
            "year_a": ya,
            "year_b": yb,
            "total_a": tot_a,
            "total_b": tot_b,
            "delta": tot_b - tot_a,
            "drivers": drivers,
            "baseline": {"mean": float(base.mean()), "lower": b_lo, "upper": b_hi},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mediation pathways (Sankey) — structural models with direct/indirect effects
# ─────────────────────────────────────────────────────────────────────────────
def _pretty_var(name: str) -> str:
    return name.replace("_latent", "").replace("_", " ").strip().title()


def _mediation_facts(
    model: Any,
    trace: Any,
    channels: list[str],
    interval: float,
) -> dict[str, Any] | None:
    """Direct/indirect effect flows for models with mediation structure.

    Duck-typed, draw-exact, two shapes:

    1. ``indirect_<channel>_via_<mediator>`` deterministics plus
       ``delta_direct_<channel>`` (the NestedMMM / StructuralNestedMMM
       convention) — path-coefficient scale (KPI units per unit saturated
       media).
    2. ``proportion_mediated_<channel>`` plus ``beta_<channel>`` and
       ``satsum_<channel>`` (the survey-mediation garden convention) —
       full-window incremental-KPI scale.

    Returns ``None`` when the model exposes neither, so plain MMMs never grow
    a pathways section.
    """
    post = getattr(trace, "posterior", None)
    if post is None:
        return None
    names = list(getattr(post, "data_vars", {}))
    y_std = float(getattr(model, "y_std", 1.0) or 1.0)
    outcome = "KPI"
    try:
        outcome = str(model.mff_config.kpi.name)
    except Exception:  # noqa: BLE001
        pass

    def link(src: str, dst: str, draws: np.ndarray, kind: str) -> dict[str, Any]:
        lo, hi = _eti(draws, interval)
        return {
            "source": src,
            "target": dst,
            "mean": float(draws.mean()),
            "lower": lo,
            "upper": hi,
            "kind": kind,
        }

    links: list[dict[str, Any]] = []
    mediators: list[str] = []
    units = ""

    indirect_re = re.compile(r"^indirect_(.+)_via_(.+)$")
    indirect_vars = [(v, indirect_re.match(v)) for v in names]
    indirect_vars = [(v, m) for v, m in indirect_vars if m]
    if indirect_vars:
        units = "effect strength (KPI units per unit saturated media)"
        med_inbound: dict[str, np.ndarray] = {}
        for var, m in indirect_vars:
            ch, med = m.group(1), m.group(2)
            draws = _flat_posterior(trace, var)
            if draws is None:
                continue
            draws = draws.reshape(-1)
            med_label = _pretty_var(med)
            links.append(link(ch, med_label, draws, "indirect"))
            med_inbound[med_label] = med_inbound.get(med_label, 0.0) + draws
        for ch in channels:
            d = _flat_posterior(trace, f"delta_direct_{ch}")
            if d is not None:
                links.append(link(ch, outcome, d.reshape(-1) * y_std, "direct"))
        for med_label, inbound in med_inbound.items():
            mediators.append(med_label)
            links.append(link(med_label, outcome, inbound, "mediated"))
    else:
        prop_channels = [ch for ch in channels if f"proportion_mediated_{ch}" in names]
        if not prop_channels:
            return None
        units = f"incremental {outcome} (full window)"
        med_label = "Mediator"
        latent_names = [
            v for v in names if v.endswith("_latent") and not v.startswith(("y_", "mu"))
        ]
        if len(latent_names) == 1:
            med_label = _pretty_var(latent_names[0])
        med_inbound_total: np.ndarray | None = None
        for ch in channels:
            beta = _flat_posterior(trace, f"beta_{ch}")
            satsum = _flat_posterior(trace, f"satsum_{ch}")
            if beta is None or satsum is None:
                continue
            total = beta.reshape(-1) * satsum.reshape(-1) * y_std
            if ch in prop_channels:
                prop = _flat_posterior(trace, f"proportion_mediated_{ch}")
                prop = np.clip(prop.reshape(-1), 0.0, 1.0)
                med_d = total * prop
                links.append(link(ch, med_label, med_d, "indirect"))
                links.append(link(ch, outcome, total * (1 - prop), "direct"))
                med_inbound_total = (
                    med_d if med_inbound_total is None else med_inbound_total + med_d
                )
            else:
                links.append(link(ch, outcome, total, "direct"))
        if med_inbound_total is None:
            return None
        mediators.append(med_label)
        links.append(link(med_label, outcome, med_inbound_total, "mediated"))

    if not any(lk["kind"] == "indirect" for lk in links):
        return None
    negatives = [f"{lk['source']} → {lk['target']}" for lk in links if lk["mean"] < 0]
    return {
        "outcome": outcome,
        "mediators": mediators,
        "links": links,
        "units": units,
        "interval": interval,
        "negatives": negatives,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Latent structure (factor loadings + latent trajectories)
# ─────────────────────────────────────────────────────────────────────────────
_LATENT_EXCLUDE_NAMES = {
    "channel_contributions",
    "media_total",
    "controls_total",
    "control_contributions",
    "y_obs_scaled",
    "mu",
    "intercept_component",
    "trend_component",
    "seasonality_component",
    "geo_component",
    "product_component",
}
_LATENT_EXCLUDE_PREFIXES = (
    "beta_",
    "sat_",
    "adstock_",
    "roi_",
    "delta_",
    "gamma_",
    "sigma",
    "intercept",
    "trend",
    "season",
    "satsum_",
    "proportion_mediated_",
    "indirect_",
    "direct_",
    "loading_",
    "mu",
    "geo",
    "product",
    "u_",
    "z_",
    "w_",
    "channel_",
    "control",
    "media_",
    "y_obs",
    "tau",
    "alpha",
)


def _latent_facts(
    model: Any,
    trace: Any,
    time_idx: np.ndarray,
    n_periods: int,
    interval: float,
    max_trajectories: int = 4,
) -> dict[str, Any] | None:
    """Latent-structure facts: factor loadings (duck-typed
    ``factor_loadings_summary()``) plus latent state trajectories over time
    (posterior vars with a period- or obs-shaped axis that are not standard
    MMM parameters/components). ``None`` for plain MMMs."""
    loadings: list[dict[str, Any]] = []
    if hasattr(model, "factor_loadings_summary"):
        try:
            df = model.factor_loadings_summary()
            for rec in df.reset_index().to_dict("records"):
                mean = rec.get("loading", rec.get("mean", rec.get("value")))
                if mean is None:
                    continue
                loadings.append(
                    {
                        "indicator": str(
                            rec.get("indicator") or rec.get("item") or "?"
                        ),
                        "factor": str(rec.get("factor") or "Factor"),
                        "mean": float(mean),
                        "lower": _jsafe(rec.get("hdi_low", rec.get("lower"))),
                        "upper": _jsafe(rec.get("hdi_high", rec.get("upper"))),
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"interactive report: factor loadings unavailable: {e}")

    trajectories: list[dict[str, Any]] = []
    post = getattr(trace, "posterior", None)
    n_obs = int(time_idx.shape[0])
    if post is not None:
        onehot = _period_onehot(time_idx, n_periods)
        counts = np.maximum(onehot.sum(axis=0), 1.0)
        lo_q = (1.0 - interval) / 2.0 * 100.0
        for name in getattr(post, "data_vars", {}):
            if name in _LATENT_EXCLUDE_NAMES or name.startswith(
                _LATENT_EXCLUDE_PREFIXES
            ):
                continue
            vals = _flat_posterior(trace, name)
            if vals is None or vals.ndim != 2:
                continue
            if vals.shape[1] == n_periods:
                per = vals
            elif vals.shape[1] == n_obs and n_obs != n_periods:
                per = (vals @ onehot) / counts
            elif vals.shape[1] == n_obs:
                per = vals
            else:
                continue
            trajectories.append(
                {
                    "name": _pretty_var(name),
                    "median": _jsafe(np.median(per, axis=0)),
                    "lower": _jsafe(np.percentile(per, lo_q, axis=0)),
                    "upper": _jsafe(np.percentile(per, 100.0 - lo_q, axis=0)),
                }
            )
            if len(trajectories) >= max_trajectories:
                break

    if not loadings and not trajectories:
        return None
    return {
        "loadings": loadings,
        "trajectories": trajectories,
        "interval": interval,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity battery (no refits — window/data perturbations + estimator swap)
# ─────────────────────────────────────────────────────────────────────────────
def _window_specs(
    periods: list[str],
    n_periods: int,
    spend: dict[str, np.ndarray],
    channels: list[str],
) -> list[dict[str, Any]]:
    """Alternative analysis windows. Each spec carries per-channel period
    masks (most share one mask; the top-spend exclusion is channel-specific)."""
    full = np.ones(n_periods, dtype=bool)
    specs: list[dict[str, Any]] = [
        {"label": "Base (full window)", "masks": {ch: full for ch in channels}}
    ]
    if n_periods >= 26:
        m = full.copy()
        m[-13:] = False
        specs.append(
            {"label": "Excl. last quarter", "masks": {ch: m for ch in channels}}
        )
    half = n_periods // 2
    if half >= 8:
        first = np.zeros(n_periods, dtype=bool)
        first[:half] = True
        specs.append({"label": "First half", "masks": {ch: first for ch in channels}})
        specs.append({"label": "Second half", "masks": {ch: ~first for ch in channels}})

    top_masks: dict[str, np.ndarray] = {}
    for ch in channels:
        s = spend.get(ch)
        m = full.copy()
        if s is not None and np.isfinite(s).any() and (s > 0).sum() > 10:
            m[np.argsort(s)[-5:]] = False
        top_masks[ch] = m
    specs.append({"label": "Excl. top-5 spend wks", "masks": top_masks})

    years = [p[:4] for p in periods]
    counts: dict[str, int] = {}
    for y in years:
        counts[y] = counts.get(y, 0) + 1
    full_years = [y for y, c in sorted(counts.items()) if c >= 26]
    if len(full_years) >= 2:
        for y in full_years[-2:]:
            m = np.array([yy != y for yy in years])
            specs.append({"label": f"Excl. {y}", "masks": {ch: m for ch in channels}})
    return specs


def _sensitivity_facts(
    model: Any,
    channels: list[str],
    contrib_dp: dict[str, np.ndarray],
    spend: dict[str, np.ndarray],
    divisor_meta: dict[str, dict[str, Any]],
    periods: list[str],
    n_periods: int,
    interval: float,
    include_counterfactual: bool,
    random_seed: int | None,
) -> dict[str, Any]:
    specs = _window_specs(periods, n_periods, spend, channels)
    labels = [s["label"] for s in specs]
    series: dict[str, dict[str, list]] = {
        ch: {"mean": [], "lower": [], "upper": []} for ch in channels
    }
    for spec in specs:
        for ch in channels:
            mask = spec["masks"][ch]
            denom = float(np.nansum(spend[ch][mask]))
            draws_pt = contrib_dp[ch][:, mask]
            if denom <= 0 or not draws_pt.size:
                series[ch]["mean"].append(None)
                series[ch]["lower"].append(None)
                series[ch]["upper"].append(None)
                continue
            roi = np.nansum(draws_pt, axis=1) / denom
            lo, hi = _eti(roi, interval)
            series[ch]["mean"].append(float(roi.mean()))
            series[ch]["lower"].append(lo)
            series[ch]["upper"].append(hi)

    notes: list[str] = [
        "Window specs re-aggregate the same posterior draws over sub-windows; "
        "they test stability of the estimate, not model misspecification "
        "outside the fitted family."
    ]
    if include_counterfactual:
        try:
            cf = model.compute_counterfactual_contributions(
                compute_uncertainty=True, random_seed=random_seed
            )
            labels.append("Zero-out counterfactual")
            for ch in channels:
                denom = float(np.nansum(spend[ch]))
                if denom <= 0 or ch not in cf.total_contributions.index:
                    series[ch]["mean"].append(None)
                    series[ch]["lower"].append(None)
                    series[ch]["upper"].append(None)
                    continue
                series[ch]["mean"].append(float(cf.total_contributions[ch]) / denom)
                lo = cf.contribution_hdi_low
                hi = cf.contribution_hdi_high
                series[ch]["lower"].append(
                    float(lo[ch]) / denom if lo is not None else None
                )
                series[ch]["upper"].append(
                    float(hi[ch]) / denom if hi is not None else None
                )
            notes.append(
                "The zero-out counterfactual re-estimates each channel by "
                "switching it off and re-predicting — a genuinely different "
                "estimator of the same estimand."
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"interactive report: counterfactual spec skipped: {e}")

    references = {ch: divisor_meta.get(ch, {}).get("reference", 1.0) for ch in channels}
    return {
        "estimand_label": "Contribution ROI (full-window unless noted)",
        "specs": labels,
        "series": {ch: series[ch] for ch in channels},
        "references": references,
        "interval": interval,
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Headline numbers (Python-side, for insights + tests)
# ─────────────────────────────────────────────────────────────────────────────
def _evidence_facts(model: Any, channels: list[str]) -> dict[str, dict[str, Any]]:
    """Per-channel evidence tier + identifiability flag (issue #102), computed on
    the SAME three signals as the classic/augur reports so the trust language is
    consistent across every deliverable. Best-effort — any failure yields ``{}``.
    """
    try:
        from ..evidence import channel_evidence, collinearity_from_matrix

        exp: set[str] = set()
        for e in getattr(model, "experiments", None) or []:
            ch = getattr(e, "channel", None)
            if ch is not None:
                exp.add(str(ch))

        learning = None
        fn = getattr(model, "compute_parameter_learning", None)
        if callable(fn) and getattr(model, "_trace", None) is not None:
            try:
                learning = fn(prior_samples=400, random_seed=0)
            except Exception:  # noqa: BLE001 — best-effort
                learning = None

        collinearity = None
        X = getattr(model, "X_media_raw", None)
        if X is None:
            X = getattr(model, "X_media", None)
        try:
            Xm = np.asarray(X, dtype=float)
            if Xm.ndim == 2 and Xm.shape[1] == len(channels) and Xm.shape[0] >= 3:
                collinearity = collinearity_from_matrix(Xm, channels)
        except (TypeError, ValueError):
            collinearity = None

        ev = channel_evidence(
            channels,
            experiment_channels=exp,
            learning=learning,
            collinearity=collinearity,
        )
        return {ch: e.to_dict() for ch, e in ev.items()}
    except Exception:  # noqa: BLE001 — the report renders fine without evidence
        return {}


def _headline_facts(
    channels: list[str],
    actual_national: np.ndarray,
    contrib_dp: dict[str, np.ndarray],
    spend: dict[str, np.ndarray],
    divisor_meta: dict[str, dict[str, Any]],
    fit: dict[str, Any],
    interval: float,
    evidence: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total_kpi = float(np.nansum(actual_national))
    media_draws = np.sum([contrib_dp[ch].sum(axis=1) for ch in channels], axis=0)
    m_lo, m_hi = _eti(media_draws, interval)

    monetary = [ch for ch in channels if divisor_meta.get(ch, {}).get("is_monetary")]
    blended = None
    if monetary:
        num = np.sum([contrib_dp[ch].sum(axis=1) for ch in monetary], axis=0)
        den = float(np.sum([np.nansum(spend[ch]) for ch in monetary]))
        if den > 0:
            b_lo, b_hi = _eti(num / den, interval)
            blended = {
                "mean": float((num / den).mean()),
                "lower": b_lo,
                "upper": b_hi,
                "spend": den,
            }

    share = media_draws / total_kpi if total_kpi > 0 else None
    channel_rows = []
    for ch in channels:
        denom = float(np.nansum(spend[ch]))
        meta = divisor_meta.get(ch, {})
        if denom <= 0:
            continue
        roi = contrib_dp[ch].sum(axis=1) / denom
        lo, hi = _eti(roi, interval)
        channel_rows.append(
            {
                "channel": ch,
                "roi_mean": float(roi.mean()),
                "roi_lower": lo,
                "roi_upper": hi,
                "spend": denom,
                "is_monetary": bool(meta.get("is_monetary", True)),
                "label": meta.get("roi_label") or "ROI",
                "reference": meta.get("reference", 1.0),
                # Evidence tier + identifiability flag (issue #102), so the
                # recompute-in-browser forest / deep-dive can chip every number.
                "evidence": (evidence or {}).get(ch),
            }
        )

    nat_stats = (fit.get("series", {}).get("National") or {}).get("stats", {})
    out: dict[str, Any] = {
        "total_kpi": total_kpi,
        "media_total": {
            "mean": float(media_draws.mean()),
            "lower": m_lo,
            "upper": m_hi,
        },
        "blended_roi": blended,
        "channels": channel_rows,
        "fit": nat_stats,
        "interval": interval,
    }
    if share is not None:
        s_lo, s_hi = _eti(share, interval)
        out["media_share"] = {
            "mean": float(share.mean()),
            "lower": s_lo,
            "upper": s_hi,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def interactive_report_facts(
    model: Any,
    results: Any = None,
    *,
    max_draws: int = 200,
    curve_max_draws: int = 120,
    curve_multipliers: tuple[float, ...] | None = None,
    marginal_bump_pct: float = 5.0,
    interval: float = 0.90,
    n_prior_samples: int = 300,
    include_prior_sections: bool = True,
    include_counterfactual_spec: bool = True,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute every fact the interactive MMM Results Report embeds.

    Parameters largely trade fidelity for build time / payload size:
    ``max_draws`` thins the posterior for the embedded per-draw matrices,
    ``curve_max_draws`` for the response-curve grid (one posterior pass per
    grid level), and ``include_counterfactual_spec`` adds one zero-out
    counterfactual pass per channel to the sensitivity battery.
    """
    trace = getattr(model, "_trace", None) or getattr(results, "trace", None)
    if trace is None:
        raise ValueError("interactive_report_facts requires a fitted model.")

    channels = [str(c) for c in getattr(model, "channel_names", [])]
    time_idx = np.asarray(model.time_idx, dtype=int)
    periods_index = _resolve_periods(model)
    n_periods = len(periods_index)
    periods = [str(p)[:10] for p in periods_index]
    onehot = _period_onehot(time_idx, n_periods)

    actual_national = np.asarray(model.y_raw, dtype=float) @ onehot

    # Posterior-predictive fit (bands + stats, per geo + national) and the
    # replicated-vs-observed test statistics (one shared predict() pass).
    fit, ppc_stats = _fit_facts(model, time_idx, n_periods, random_seed)

    # Per-draw per-period contributions in KPI units (base + paired bump).
    contrib_doc = model.sample_channel_contributions(
        max_draws=max_draws, random_seed=random_seed
    )  # (D, n_obs, C)
    bump = 1.0 + marginal_bump_pct / 100.0
    contrib_bump_doc = model.sample_channel_contributions(
        X_media=np.asarray(model.X_media_raw, dtype=float) * bump,
        max_draws=max_draws,
        random_seed=random_seed,
    )
    contrib_dpc = np.einsum("doc,op->dpc", contrib_doc, onehot)
    delta_dpc = np.einsum("doc,op->dpc", contrib_bump_doc - contrib_doc, onehot)
    n_draws = int(contrib_dpc.shape[0])
    contrib_dp = {ch: contrib_dpc[:, :, i] for i, ch in enumerate(channels)}
    delta_dp = {ch: delta_dpc[:, :, i] for i, ch in enumerate(channels)}

    # Spend-equivalent (divisor) series per channel + metric meta.
    spend, divisor_meta = _divisor_series(model, channels, time_idx, n_periods)

    # Response-curve grid: one posterior pass per multiplier level.
    multipliers = [float(m) for m in (curve_multipliers or DEFAULT_CURVE_MULTIPLIERS)]
    x_raw = np.asarray(model.X_media_raw, dtype=float)
    level_totals: list[np.ndarray] = []
    curve_draws_n: int | None = None
    for m in multipliers:
        if m == 0.0:
            level_totals.append(None)  # filled after we know the draw count
            continue
        tot = model.sample_channel_contributions(
            X_media=x_raw * m, max_draws=curve_max_draws, random_seed=random_seed
        ).sum(
            axis=1
        )  # (D, C)
        level_totals.append(tot)
        curve_draws_n = tot.shape[0]
    if curve_draws_n is None:
        curve_draws_n = 1
    level_totals = [
        np.zeros((curve_draws_n, len(channels))) if t is None else t
        for t in level_totals
    ]
    curve_stack = np.stack(level_totals, axis=-1)  # (D, C, L)
    curves = {
        "multipliers": multipliers,
        "n_draws": int(curve_draws_n),
        "draws_b64": {
            ch: _b64_f32(curve_stack[:, i, :]) for i, ch in enumerate(channels)
        },
        "spend_total": {ch: float(np.nansum(spend[ch])) for ch in channels},
        "n_periods": n_periods,
    }

    # Carryover kernels from the posterior adstock parameters.
    carryover = _carryover_facts(model, trace, channels, interval)

    # Prior sections (prior predictive fan + prior estimand draws).
    ppc_prior: dict[str, Any] | None = None
    prior_est: dict[str, Any] = {}
    prior_idata: Any = None
    if include_prior_sections:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prior_idata = sample_prior(model, n_prior_samples, random_seed)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"interactive report: prior sampling unavailable: {e}")
        if prior_idata is not None:
            try:
                ppc_prior = _jsafe(prior_predictive_facts(model, prior_idata))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"interactive report: prior predictive unavailable: {e}")
            try:
                prior_est = prior_estimand_facts(model, prior_idata)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"interactive report: prior estimands unavailable: {e}")

    posterior_roi_draws = {
        ch: (
            contrib_dp[ch].sum(axis=1) / float(np.nansum(spend[ch]))
            if float(np.nansum(spend[ch])) > 0
            else None
        )
        for ch in channels
    }
    prior_posterior = _prior_posterior_rows(
        channels,
        prior_est,
        {k: v for k, v in posterior_roi_draws.items() if v is not None},
        divisor_meta,
        interval,
    )

    sensitivity = _sensitivity_facts(
        model,
        channels,
        contrib_dp,
        spend,
        divisor_meta,
        periods,
        n_periods,
        interval,
        include_counterfactual_spec,
        random_seed,
    )

    evidence = _evidence_facts(model, channels)
    headline = _headline_facts(
        channels,
        actual_national,
        contrib_dp,
        spend,
        divisor_meta,
        fit,
        interval,
        evidence,
    )

    yoy = _yoy_facts(actual_national, contrib_dp, periods, channels, interval)

    mediation = _mediation_facts(model, trace, channels, interval)
    latent = _latent_facts(model, trace, time_idx, n_periods, interval)

    assumptions = [r.to_dict() for r in model_assumptions(model)]

    diagnostics: dict[str, Any] = {}
    if results is not None:
        diagnostics = dict(getattr(results, "diagnostics", {}) or {})
        diagnostics["approximate"] = bool(getattr(results, "approximate", False))
        converged = getattr(results, "converged", None)
        diagnostics["converged"] = None if converged is None else bool(converged)

    meta: dict[str, Any] = {
        "kpi": None,
        "channels": channels,
        "geos": fit["order"],
        "n_periods": n_periods,
        "n_draws": n_draws,
        "date_start": periods[0] if periods else None,
        "date_end": periods[-1] if periods else None,
        "interval": interval,
        "marginal_bump_pct": marginal_bump_pct,
        "fit_method": diagnostics.get("fit_method"),
        "approximate": bool(diagnostics.get("approximate", False)),
    }
    try:
        meta["kpi"] = str(model.mff_config.kpi.name)
    except Exception:  # noqa: BLE001
        pass
    if meta["kpi"] is None:
        # Extension models carry no MFF config; name the KPI from the primary
        # outcome when the model is multi-outcome (Multivariate / Combined).
        names = getattr(model, "outcome_names", None)
        if names:
            try:
                idx = model._primary_outcome_index()
            except Exception:  # noqa: BLE001
                idx = 0
            if 0 <= idx < len(names):
                meta["kpi"] = str(names[idx])

    return {
        "meta": meta,
        "periods": periods,
        "actual_national": _jsafe(actual_national),
        "fit": fit,
        "ppc_stats": ppc_stats,
        "contrib": {
            "n_draws": n_draws,
            "draws_b64": {ch: _b64_f32(contrib_dp[ch]) for ch in channels},
        },
        "marginal": {
            "bump_pct": marginal_bump_pct,
            "draws_b64": {ch: _b64_f32(delta_dp[ch]) for ch in channels},
        },
        "spend": {ch: _jsafe(spend[ch]) for ch in channels},
        "divisor_meta": _jsafe(divisor_meta),
        "curves": curves,
        "carryover": carryover,
        "prior_posterior": {"interval": interval, "rows": prior_posterior},
        "sensitivity": sensitivity,
        "ppc_prior": ppc_prior,
        "headline": headline,
        "evidence": _jsafe(evidence),
        "yoy": yoy,
        "mediation": mediation,
        "latent": latent,
        "assumptions": assumptions,
        "diagnostics": _jsafe(diagnostics),
    }
