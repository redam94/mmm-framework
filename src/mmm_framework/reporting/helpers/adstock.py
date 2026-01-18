"""
Adstock computation functions for MMM reporting.

Functions for computing adstock decay weights and carryover effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from .results import AdstockResult
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
)


def compute_adstock_weights(
    model: Any,
    channels: list[str] | None = None,
    hdi_prob: float = 0.94,
) -> dict[str, AdstockResult]:
    """
    Compute adstock decay weights for each channel.

    Shows how advertising effects decay over time (carryover effects).

    Parameters
    ----------
    model : BayesianMMM
        Fitted model
    channels : list[str], optional
        Channels to compute. If None, uses all.
    hdi_prob : float
        HDI probability

    Returns
    -------
    dict[str, AdstockResult]
        Adstock curves by channel

    Examples
    --------
    >>> adstock = compute_adstock_weights(mmm)
    >>> for ch, result in adstock.items():
    ...     print(f"{ch}: half-life = {result.half_life:.1f} periods")
    """
    _check_model_fitted(model)

    posterior = _get_posterior(model)

    if channels is None:
        channels = _get_channel_names(model)

    results = {}

    for channel in channels:
        # Get alpha parameter
        alpha_samples = _get_adstock_alpha(posterior, channel)

        if alpha_samples is None:
            logger.warning(f"No adstock parameter found for {channel}")
            continue

        # Get l_max
        l_max = _get_adstock_lmax(model, channel)

        # Compute decay weights using mean alpha
        alpha_mean = float(np.mean(alpha_samples))
        alpha_lower, alpha_upper = _compute_hdi(alpha_samples, hdi_prob)

        lags = np.arange(l_max)
        weights = alpha_mean**lags
        weights = weights / weights.sum()  # Normalize

        # Half-life calculation
        if alpha_mean > 0 and alpha_mean < 1:
            half_life = np.log(0.5) / np.log(alpha_mean)
        else:
            half_life = 0.0

        # Total carryover (sum of weights beyond t=0)
        total_carryover = float(weights[1:].sum())

        results[channel] = AdstockResult(
            channel=channel,
            decay_weights=weights,
            alpha_mean=alpha_mean,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            half_life=float(half_life),
            total_carryover=total_carryover,
            l_max=l_max,
        )

    return results


def _get_adstock_alpha(posterior: Any, channel: str) -> np.ndarray | None:
    """Extract adstock alpha parameter for a channel."""
    if posterior is None:
        return None

    for prefix in ["adstock_", "alpha_", "decay_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            return _flatten_samples(posterior[name].values)

    return None


def _get_adstock_lmax(model: Any, channel: str) -> int:
    """Get l_max for a channel's adstock."""
    # Try from panel config
    if hasattr(model, "panel") and model.panel is not None:
        if hasattr(model.panel, "mff_config"):
            for mc in model.panel.mff_config.media_channels:
                if mc.name == channel:
                    return mc.adstock_lmax or 8

    # Try from model attribute
    if hasattr(model, "adstock_lmax"):
        return model.adstock_lmax

    # Default
    return 8


__all__ = [
    "compute_adstock_weights",
    "_get_adstock_alpha",
    "_get_adstock_lmax",
]
