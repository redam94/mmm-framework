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
) -> dict[str, Any]:
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
    return {"series": series, "order": order, "band_levels": list(FIT_BAND_LEVELS)}


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
        row: dict[str, Any] = {
            "channel": ch,
            "label": meta.get("roi_label") or "ROI",
            "reference": meta.get("reference", 1.0),
            "grid": _jsafe(grid),
            "posterior": {
                "density": _jsafe(_kde_curve(post, grid)),
                "mean": float(post.mean()),
                "lower": post_lo,
                "upper": post_hi,
            },
        }
        if prior_draws.size:
            pr_lo, pr_hi = _eti(prior_draws, interval)
            row["prior"] = {
                "density": _jsafe(_kde_curve(prior_draws, grid)),
                "mean": float(prior_draws.mean()),
                "lower": pr_lo,
                "upper": pr_hi,
            }
        rows.append(row)
    return rows


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
def _headline_facts(
    channels: list[str],
    actual_national: np.ndarray,
    contrib_dp: dict[str, np.ndarray],
    spend: dict[str, np.ndarray],
    divisor_meta: dict[str, dict[str, Any]],
    fit: dict[str, Any],
    interval: float,
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
    periods_index = model.panel.coords.periods
    n_periods = len(periods_index)
    periods = [str(p)[:10] for p in periods_index]
    onehot = _period_onehot(time_idx, n_periods)

    actual_national = np.asarray(model.y_raw, dtype=float) @ onehot

    # Posterior-predictive fit (bands + stats, per geo + national).
    fit = _fit_facts(model, time_idx, n_periods, random_seed)

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

    headline = _headline_facts(
        channels, actual_national, contrib_dp, spend, divisor_meta, fit, interval
    )

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

    return {
        "meta": meta,
        "periods": periods,
        "actual_national": _jsafe(actual_national),
        "fit": fit,
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
        "assumptions": assumptions,
        "diagnostics": _jsafe(diagnostics),
    }
