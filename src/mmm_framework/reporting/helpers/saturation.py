"""
Saturation curve computation functions for MMM reporting.

Functions for computing saturation curves with uncertainty quantification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from .results import SaturationCurveResult
from .roi import _extract_spend_from_model
from .utils import (
    _check_model_fitted,
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

        # Compute response curves
        response_samples = np.zeros((len(beta_samples), n_points))
        for i in range(n_points):
            saturated = _apply_saturation(spend_grid[i], sat_params)
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

    # Try Hill saturation (kappa, slope)
    kappa_name = None
    slope_name = None
    for prefix in ["kappa_", "K_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            kappa_name = name
            break
    for prefix in ["slope_", "S_", "n_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            slope_name = name
            break

    if kappa_name and slope_name:
        params["type"] = "hill"
        params["kappa"] = _flatten_samples(posterior[kappa_name].values)
        params["slope"] = _flatten_samples(posterior[slope_name].values)
        return params

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
        return x**slope / (kappa**slope + x**slope)

    elif sat_type == "logistic":
        lam = params["lam"]
        return 1 / (1 + np.exp(-lam * (x - 0.5)))

    else:
        # Linear (no saturation)
        return np.ones_like(params.get("lam", np.array([1.0]))) * x


__all__ = [
    "compute_saturation_curves_with_uncertainty",
    "_get_saturation_params",
    "_get_beta_samples",
    "_apply_saturation",
]
