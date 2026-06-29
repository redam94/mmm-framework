"""
Saturation curve computation functions for MMM reporting.

Functions for computing saturation curves with uncertainty quantification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from .results import SaturationCurveResult, SpendResponseZones
from .roi import _extract_spend_from_model
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
    _get_scaling_params,
)


def compute_saturation_curves_with_uncertainty(
    model: Any,
    channels: list[str] | None = None,
    n_points: int = 100,
    spend_multiplier: float = 1.5,
    n_samples: int = 500,
    hdi_prob: float = 0.94,
) -> dict[str, SaturationCurveResult]:
    """
    Compute saturation curves with uncertainty bands.

    Shows how channel response varies with spend level, including
    diminishing returns and current position on the curve.

    Parameters
    ----------
    model : BayesianMMM
        Fitted model
    channels : list[str], optional
        Channels to compute. If None, uses all channels.
    n_points : int
        Number of points on spend grid
    spend_multiplier : float
        Max spend as multiple of current max
    n_samples : int
        Number of posterior samples for uncertainty
    hdi_prob : float
        HDI probability

    Returns
    -------
    dict[str, SaturationCurveResult]
        Saturation curves by channel

    Examples
    --------
    >>> curves = compute_saturation_curves_with_uncertainty(mmm)
    >>> for ch, curve in curves.items():
    ...     print(f"{ch}: {curve.saturation_level:.0%} saturated")
    """
    _check_model_fitted(model)

    posterior = _get_posterior(model)
    y_mean, y_std = _get_scaling_params(model)

    if channels is None:
        channels = _get_channel_names(model)

    spend_data = _extract_spend_from_model(model)
    n_obs = getattr(model, "n_obs", 52)

    results = {}

    for channel in channels:
        # Get spend range
        total_spend = spend_data.get(channel, 100000)
        max_spend = total_spend / n_obs * spend_multiplier
        spend_grid = np.linspace(0, max_spend, n_points)

        # Get parameters
        sat_params = _get_saturation_params(model, posterior, channel)
        beta_samples = _get_beta_samples(posterior, channel)

        if sat_params is None or beta_samples is None:
            logger.warning(f"Skipping saturation curve for {channel}")
            continue

        # Subsample if needed
        if len(beta_samples) > n_samples:
            idx = np.random.choice(len(beta_samples), n_samples, replace=False)
            beta_samples = beta_samples[idx]
            sat_params = {
                k: v[idx] if isinstance(v, np.ndarray) else v
                for k, v in sat_params.items()
            }

        # Scale grid by media_max to match how model was fitted
        if hasattr(model, "_media_max") and channel in model._media_max:
            scale_factor = model._media_max[channel] + 1e-8
        else:
            scale_factor = 1.0

        scaled_grid = spend_grid / scale_factor

        # Compute response curves
        response_samples = np.zeros((len(beta_samples), n_points))
        for i in range(n_points):
            saturated = _apply_saturation(scaled_grid[i], sat_params)
            response_samples[:, i] = beta_samples * saturated * y_std

        # Compute statistics
        response_mean = response_samples.mean(axis=0)
        response_lower = np.percentile(
            response_samples, (1 - hdi_prob) / 2 * 100, axis=0
        )
        response_upper = np.percentile(
            response_samples, (1 + hdi_prob) / 2 * 100, axis=0
        )

        # Current spend position
        current_spend = total_spend / n_obs
        current_idx = np.argmin(np.abs(spend_grid - current_spend))
        current_response = float(response_mean[current_idx])

        # Saturation level (% of max response)
        max_response = float(response_mean[-1]) if response_mean[-1] > 0 else 1.0
        saturation_level = current_response / max_response if max_response > 0 else 0.0

        # Marginal response at current spend
        if current_idx > 0 and current_idx < n_points - 1:
            marginal = (
                response_mean[current_idx + 1] - response_mean[current_idx - 1]
            ) / (spend_grid[current_idx + 1] - spend_grid[current_idx - 1])
        else:
            marginal = 0.0

        results[channel] = SaturationCurveResult(
            channel=channel,
            spend_grid=spend_grid,
            response_mean=response_mean,
            response_lower=response_lower,
            response_upper=response_upper,
            current_spend=current_spend,
            current_response=current_response,
            saturation_level=saturation_level,
            marginal_response_at_current=float(marginal),
        )

    return results


def _get_saturation_params(
    model: Any,
    posterior: Any,
    channel: str,
) -> dict[str, Any] | None:
    """Extract saturation parameters for a channel."""
    if posterior is None:
        return None

    params = {}

    # Try exponential saturation (sat_lam)
    for prefix in ["sat_lam_", "saturation_lam_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            params["type"] = "exponential"
            params["lam"] = _flatten_samples(posterior[name].values)
            return params

    # Try Hill saturation (kappa, slope). ``sat_half_``/``sat_slope_`` are the
    # core BayesianMMM's per-channel Hill RVs.
    kappa_name = None
    slope_name = None
    for prefix in ["sat_half_", "kappa_", "K_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            kappa_name = name
            break
    for prefix in ["sat_slope_", "slope_", "S_", "n_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            slope_name = name
            break

    if kappa_name and slope_name:
        params["type"] = "hill"
        params["kappa"] = _flatten_samples(posterior[kappa_name].values)
        params["slope"] = _flatten_samples(posterior[slope_name].values)
        return params

    # ``sat_half_`` without a slope: the core model's michaelis_menten or tanh
    # saturation. The two are indistinguishable from RV names alone, so consult
    # the model's per-channel config; if unavailable, skip rather than plot the
    # wrong curve.
    if kappa_name is not None:
        sat_type = None
        try:
            sat_type = model._get_saturation_config(channel).type.value
        except Exception:  # non-core model or missing config
            sat_type = None
        if sat_type in ("michaelis_menten", "tanh"):
            params["type"] = sat_type
            params["kappa"] = _flatten_samples(posterior[kappa_name].values)
            return params
        logger.warning(
            f"Saturation params for {channel}: found {kappa_name} but cannot "
            "determine the saturation form (michaelis_menten vs tanh); skipping."
        )
        return None

    # Try logistic saturation
    for prefix in ["logistic_lam_", "mu_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            params["type"] = "logistic"
            params["lam"] = _flatten_samples(posterior[name].values)
            return params

    return None


def _get_beta_samples(posterior: Any, channel: str) -> np.ndarray | None:
    """Extract beta coefficient samples for a channel."""
    if posterior is None:
        return None

    for prefix in ["beta_", "beta_media_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            return _flatten_samples(posterior[name].values)

    return None


def _apply_saturation(
    x: float | np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Apply saturation function to input."""
    sat_type = params.get("type", "exponential")

    if sat_type == "exponential":
        lam = params["lam"]
        return 1 - np.exp(-lam * x)

    elif sat_type == "hill":
        kappa = params["kappa"]
        slope = params["slope"]
        x_safe = np.clip(x, 1e-9, None)
        return x_safe**slope / (kappa**slope + x_safe**slope)

    elif sat_type == "michaelis_menten":
        kappa = params["kappa"]
        return x / (x + kappa)

    elif sat_type == "tanh":
        kappa = params["kappa"]
        return np.tanh(x / kappa)

    elif sat_type == "logistic":
        lam = params["lam"]
        return 1 / (1 + np.exp(-lam * (x - 0.5)))

    else:
        # Linear (no saturation)
        return np.ones_like(params.get("lam", np.array([1.0]))) * x


def _apply_saturation_derivative(
    x: float,
    params: dict[str, Any],
) -> np.ndarray:
    """Analytic derivative ``df/dx`` of :func:`_apply_saturation` (w.r.t. the
    *normalized* input ``x``).

    Used for marginal ROI: an exact derivative avoids the noise of numerical
    differentiation and matches the saturation form bit-for-bit. ``x`` is a scalar
    (a single normalized spend level); the saturation parameters are per-posterior-
    sample arrays, so the return broadcasts to one value per sample.
    """
    sat_type = params.get("type", "exponential")

    if sat_type == "exponential":  # f = 1 - exp(-lam·x)  ->  f' = lam·exp(-lam·x)
        lam = params["lam"]
        return lam * np.exp(-lam * x)

    elif sat_type == "hill":  # f = x^n/(k^n+x^n) -> f' = n·k^n·x^(n-1)/(k^n+x^n)^2
        kappa = params["kappa"]
        slope = params["slope"]
        xs = np.clip(x, 1e-9, None)
        ks = kappa**slope
        return slope * ks * (xs ** (slope - 1)) / ((ks + xs**slope) ** 2)

    elif sat_type == "michaelis_menten":  # f = x/(x+k) -> f' = k/(x+k)^2
        kappa = params["kappa"]
        return kappa / ((x + kappa) ** 2)

    elif sat_type == "tanh":  # f = tanh(x/k) -> f' = (1/k)·(1 - tanh^2(x/k))
        kappa = params["kappa"]
        th = np.tanh(x / kappa)
        return (1.0 / kappa) * (1.0 - th**2)

    elif sat_type == "logistic":  # f = sigmoid(lam(x-0.5)) -> f' = lam·f·(1-f)
        lam = params["lam"]
        f = 1.0 / (1.0 + np.exp(-lam * (x - 0.5)))
        return lam * f * (1.0 - f)

    else:  # linear: f = x -> f' = 1
        base = params.get("lam", params.get("kappa", np.array([1.0])))
        return np.ones_like(base)


def _largest_spend_at_or_above(grid: np.ndarray, y: np.ndarray, thr: float) -> float:
    """Largest spend on ``grid`` where curve ``y`` is ≥ ``thr``, linearly
    interpolating the crossing.

    For a (generally decreasing) marginal-ROI curve this is the right edge of the
    region that is "at least this efficient" — the zone boundary. Returns 0.0 when
    even the smallest spend is already below ``thr`` (zone empty on the left), and
    the grid max when the curve never drops below ``thr`` in range.
    """
    above = np.where(y >= thr)[0]
    if len(above) == 0:
        return float(grid[0])
    j = int(above[-1])
    if j >= len(grid) - 1:
        return float(grid[-1])
    y0, y1 = float(y[j]), float(y[j + 1])
    x0, x1 = float(grid[j]), float(grid[j + 1])
    if y0 == y1:
        return x1
    t = (y0 - thr) / (y0 - y1)
    t = min(max(t, 0.0), 1.0)
    return x0 + t * (x1 - x0)


def compute_response_zones(
    model: Any,
    channels: list[str] | None = None,
    *,
    n_points: int = 60,
    spend_multiplier: float = 2.0,
    n_samples: int = 500,
    hdi_prob: float = 0.94,
    break_even: float = 1.0,
    band: float = 0.15,
) -> dict[str, SpendResponseZones]:
    """Per-channel ROI(spend) and **marginal ROI(spend)** curves with
    breakthrough / optimal / saturation spend zones, defined on marginal-ROI
    break-even bands rather than percent-of-response.

    For each channel the (per-period) response is ``β·sat(spend/scale)·y_std``;
    average ROI is ``response/spend`` and marginal ROI is the analytic derivative
    ``β·sat'(spend/scale)·(1/scale)·y_std``. The zones partition the spend axis by
    where marginal ROI sits relative to ``break_even`` (see
    :class:`SpendResponseZones`). Everything is computed directly from the fitted
    posterior — no AI calls.

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    channels : list[str], optional
        Channels to compute (default: all).
    n_points, spend_multiplier, n_samples, hdi_prob
        Spend grid resolution, max spend as a multiple of current per-period
        spend, posterior subsample size, and HDI mass.
    break_even : float
        Marginal-ROI break-even target (default 1.0 — one KPI unit per dollar;
        pass ``1/margin`` for a margin-adjusted target).
    band : float
        Fractional half-width of the "optimal" band around ``break_even``
        (default 0.15 ⇒ optimal where mROI ∈ [0.85, 1.15]·break_even).

    Returns
    -------
    dict[str, SpendResponseZones]
    """
    _check_model_fitted(model)

    posterior = _get_posterior(model)
    _y_mean, y_std = _get_scaling_params(model)
    if channels is None:
        channels = _get_channel_names(model)
    spend_data = _extract_spend_from_model(model)
    n_obs = getattr(model, "n_obs", 52) or 52

    results: dict[str, SpendResponseZones] = {}
    for channel in channels:
        total_spend = float(spend_data.get(channel, 0.0))
        if total_spend <= 0:
            continue
        current_spend = total_spend / n_obs
        max_spend = current_spend * spend_multiplier
        if max_spend <= 0:
            continue
        spend_grid = np.linspace(0.0, max_spend, n_points)

        sat_params = _get_saturation_params(model, posterior, channel)
        beta_samples = _get_beta_samples(posterior, channel)
        if sat_params is None or beta_samples is None:
            logger.warning(f"Skipping response zones for {channel}")
            continue

        if len(beta_samples) > n_samples:
            idx = np.random.choice(len(beta_samples), n_samples, replace=False)
            beta_samples = beta_samples[idx]
            sat_params = {
                k: (v[idx] if isinstance(v, np.ndarray) else v)
                for k, v in sat_params.items()
            }

        if hasattr(model, "_media_max") and channel in model._media_max:
            scale = float(model._media_max[channel]) + 1e-8
        else:
            scale = 1.0
        scaled = spend_grid / scale

        n_s = len(beta_samples)
        response_samples = np.zeros((n_s, n_points))
        mroi_samples = np.zeros((n_s, n_points))
        for i in range(n_points):
            sat = _apply_saturation(scaled[i], sat_params)
            dsat = _apply_saturation_derivative(scaled[i], sat_params)
            response_samples[:, i] = beta_samples * sat * y_std
            mroi_samples[:, i] = beta_samples * dsat * (1.0 / scale) * y_std

        # Average ROI = response / spend; the s→0 limit is the marginal ROI there.
        roi_samples = np.empty((n_s, n_points))
        roi_samples[:, 0] = mroi_samples[:, 0]
        roi_samples[:, 1:] = response_samples[:, 1:] / spend_grid[1:]

        def _stat(arr: np.ndarray):
            return (
                arr.mean(axis=0),
                np.percentile(arr, (1 - hdi_prob) / 2 * 100, axis=0),
                np.percentile(arr, (1 + hdi_prob) / 2 * 100, axis=0),
            )

        response_mean, response_lower, response_upper = _stat(response_samples)
        roi_mean, roi_lower, roi_upper = _stat(roi_samples)
        mroi_mean, mroi_lower, mroi_upper = _stat(mroi_samples)

        # Current ROI/mROI sampled exactly at current_spend (not grid-snapped).
        cs = current_spend / scale
        cur_resp = beta_samples * _apply_saturation(cs, sat_params) * y_std
        cur_mroi = (
            beta_samples
            * _apply_saturation_derivative(cs, sat_params)
            * (1.0 / scale)
            * y_std
        )
        cur_roi = cur_resp / current_spend if current_spend > 0 else cur_mroi
        current_response = float(cur_resp.mean())
        current_roi = float(cur_roi.mean())
        current_roi_hdi = _compute_hdi(cur_roi, hdi_prob)
        current_mroi = float(cur_mroi.mean())
        current_mroi_hdi = _compute_hdi(cur_mroi, hdi_prob)

        # Zones from the posterior-mean mROI curve.
        t_hi = break_even * (1.0 + band)
        t_lo = break_even * (1.0 - band)
        gmax = float(spend_grid[-1])
        s_bt_opt = _largest_spend_at_or_above(spend_grid, mroi_mean, t_hi)
        s_opt_sat = _largest_spend_at_or_above(spend_grid, mroi_mean, t_lo)
        s_opt_sat = max(s_opt_sat, s_bt_opt)  # keep ordering under noise
        opt_spend_val = _largest_spend_at_or_above(spend_grid, mroi_mean, break_even)
        # Only a meaningful optimum if mROI actually crosses break-even in range.
        optimal_spend = (
            float(opt_spend_val)
            if (mroi_mean[0] >= break_even >= mroi_mean[-1])
            else None
        )
        optimal_roi = None
        if optimal_spend is not None:
            oi = int(np.argmin(np.abs(spend_grid - optimal_spend)))
            optimal_roi = float(roi_mean[oi])

        if current_mroi >= t_hi:
            current_zone, recommendation = "breakthrough", "increase"
        elif current_mroi >= t_lo:
            current_zone, recommendation = "optimal", "hold"
        else:
            current_zone, recommendation = "saturation", "reduce"

        headroom = (
            float(optimal_spend - current_spend) if optimal_spend is not None else None
        )

        results[channel] = SpendResponseZones(
            channel=channel,
            spend_grid=spend_grid,
            response_mean=response_mean,
            response_lower=response_lower,
            response_upper=response_upper,
            roi_mean=roi_mean,
            roi_lower=roi_lower,
            roi_upper=roi_upper,
            mroi_mean=mroi_mean,
            mroi_lower=mroi_lower,
            mroi_upper=mroi_upper,
            current_spend=current_spend,
            current_response=current_response,
            current_roi=current_roi,
            current_roi_hdi=current_roi_hdi,
            current_mroi=current_mroi,
            current_mroi_hdi=current_mroi_hdi,
            break_even=break_even,
            band=band,
            breakthrough_range=(0.0, s_bt_opt),
            optimal_range=(s_bt_opt, s_opt_sat),
            saturation_range=(s_opt_sat, gmax),
            optimal_spend=optimal_spend,
            optimal_roi=optimal_roi,
            current_zone=current_zone,
            recommendation=recommendation,
            headroom_to_optimal=headroom,
        )

    return results


__all__ = [
    "compute_saturation_curves_with_uncertainty",
    "compute_response_zones",
    "_get_saturation_params",
    "_get_beta_samples",
    "_apply_saturation",
    "_apply_saturation_derivative",
]
