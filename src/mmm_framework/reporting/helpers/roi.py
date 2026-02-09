"""
ROI computation functions for MMM reporting.

Functions for computing ROI with uncertainty quantification from fitted models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .results import ROIResult
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
    _get_scaling_params,
)


def compute_roi_with_uncertainty(
    model: Any,
    spend_data: dict[str, float] | pd.Series | None = None,
    hdi_prob: float = 0.94,
    n_samples: int | None = None,
) -> pd.DataFrame:
    """
    Compute ROI with full uncertainty quantification.

    Computes average ROI (contribution / spend) for each channel with
    credible intervals derived from the posterior distribution.

    Parameters
    ----------
    model : BayesianMMM or ExtendedMMM
        Fitted MMM model with trace
    spend_data : dict or pd.Series, optional
        Channel spend totals. If None, extracts from model's panel data.
    hdi_prob : float
        Probability mass for HDI (default 0.94)
    n_samples : int, optional
        Number of posterior samples to use. If None, uses all.

    Returns
    -------
    pd.DataFrame
        DataFrame with ROI metrics per channel including:
        - spend: Total channel spend
        - contribution_mean/lower/upper: Revenue contribution with HDI
        - roi_mean/lower/upper: ROI with HDI
        - prob_positive: P(ROI > 0)
        - prob_profitable: P(ROI > 1)

    Examples
    --------
    >>> roi_df = compute_roi_with_uncertainty(mmm)
    >>> print(roi_df[['channel', 'roi_mean', 'roi_hdi_low', 'roi_hdi_high', 'prob_profitable']])
    """
    _check_model_fitted(model)

    posterior = _get_posterior(model)
    channels = _get_channel_names(model)
    y_mean, y_std = _get_scaling_params(model)

    # Get spend data
    if spend_data is None:
        spend_data = _extract_spend_from_model(model)
    elif isinstance(spend_data, pd.Series):
        spend_data = spend_data.to_dict()

    results = []

    for channel in channels:
        # Get contribution samples
        contrib_samples = _get_contribution_samples(
            model, posterior, channel, y_mean, y_std
        )

        if contrib_samples is None or len(contrib_samples) == 0:
            logger.warning(f"No contribution samples found for {channel}")
            continue

        # Subsample if requested
        if n_samples is not None and len(contrib_samples) > n_samples:
            idx = np.random.choice(len(contrib_samples), n_samples, replace=False)
            contrib_samples = contrib_samples[idx]

        # Get spend
        spend = spend_data.get(channel, 0.0)
        if spend <= 0:
            logger.warning(f"No spend data for {channel}, skipping ROI computation")
            continue

        # Compute ROI samples
        roi_samples = contrib_samples / spend

        # Compute statistics
        contrib_mean = float(np.mean(contrib_samples))
        contrib_lower, contrib_upper = _compute_hdi(contrib_samples, hdi_prob)

        roi_mean = float(np.mean(roi_samples))
        roi_lower, roi_upper = _compute_hdi(roi_samples, hdi_prob)

        prob_positive = float(np.mean(roi_samples > 0))
        prob_profitable = float(np.mean(roi_samples > 1))

        results.append(
            ROIResult(
                channel=channel,
                spend=spend,
                contribution_mean=contrib_mean,
                contribution_lower=contrib_lower,
                contribution_upper=contrib_upper,
                roi_mean=roi_mean,
                roi_lower=roi_lower,
                roi_upper=roi_upper,
                prob_positive=prob_positive,
                prob_profitable=prob_profitable,
            )
        )

    return pd.DataFrame([r.to_dict() for r in results])


def _extract_spend_from_model(model: Any) -> dict[str, float]:
    """Extract total spend per channel from model's panel data."""
    spend = {}
    channels = _get_channel_names(model)

    def _get_column_sum(data, col_idx: int, col_name: str) -> float | None:
        """Safely get sum of a column from DataFrame or array."""
        if data is None:
            return None
        try:
            if hasattr(data, "columns") and col_name in data.columns:
                # DataFrame with matching column name
                return float(data[col_name].sum())
            elif hasattr(data, "iloc"):
                # DataFrame - use iloc
                if col_idx < data.shape[1]:
                    return float(data.iloc[:, col_idx].sum())
            elif hasattr(data, "values"):
                # Has .values attribute (DataFrame-like)
                arr = data.values
                if col_idx < arr.shape[1]:
                    return float(arr[:, col_idx].sum())
            else:
                # Numpy array
                if col_idx < data.shape[1]:
                    return float(data[:, col_idx].sum())
        except Exception:
            pass
        return None

    # Try panel data
    X_media = None
    if hasattr(model, "panel") and model.panel is not None:
        X_media = getattr(model.panel, "X_media", None)

    if X_media is None:
        X_media = getattr(model, "X_media_raw", None)

    if X_media is None:
        X_media = getattr(model, "X_media", None)

    if X_media is not None:
        for i, ch in enumerate(channels):
            val = _get_column_sum(X_media, i, ch)
            if val is not None:
                spend[ch] = val

    return spend


def _get_contribution_samples(
    model: Any,
    posterior: Any,
    channel: str,
    y_mean: float,
    y_std: float,
) -> np.ndarray | None:
    """Extract contribution samples for a channel."""
    if posterior is None:
        return None

    channels = _get_channel_names(model)

    # Try different variable naming conventions
    possible_names = [
        f"contribution_{channel}",
        f"channel_contribution_{channel}",
        f"media_contribution_{channel}",
    ]

    for var_name in possible_names:
        if var_name in posterior:
            # ALWAYS get .values before any operations
            arr = posterior[var_name].values
            samples = _flatten_samples(arr)
            if samples.ndim > 1:
                samples = samples.sum(axis=-1)
            return samples * y_std

    # Fall back to channel_contributions with index
    if "channel_contributions" in posterior:
        try:
            ch_idx = channels.index(channel)

            # Get the DataArray
            da = posterior["channel_contributions"]

            # Method 1: Try dimension-aware selection (preferred for xarray)
            if hasattr(da, "dims"):
                dims = da.dims
                logger.debug(f"channel_contributions dims: {dims}")

                # If there's a channel dimension, use .isel or .sel
                if "channel" in dims:
                    # Use integer index
                    arr = da.isel(channel=ch_idx).values
                elif len(dims) > 2:
                    # Assume last dim is channel: (chain, draw, time, channel) or (chain, draw, channel)
                    arr = da.values  # Get numpy FIRST
                    arr = _flatten_samples(arr)
                    if arr.ndim > 1 and ch_idx < arr.shape[-1]:
                        arr = arr[..., ch_idx]
                else:
                    arr = da.values
                    arr = _flatten_samples(arr)
            else:
                # No dims attribute, just get values
                arr = da.values
                arr = _flatten_samples(arr)
                if arr.ndim > 1 and ch_idx < arr.shape[-1]:
                    arr = arr[..., ch_idx]

            # Sum over time if still multidimensional
            if arr.ndim > 1:
                arr = arr.sum(axis=-1)

            return arr * y_std

        except Exception as e:
            logger.warning(
                f"Failed to extract channel_contributions for {channel}: {e}"
            )

    # Fall back to beta * media
    for beta_name in [f"beta_{channel}", f"beta_media_{channel}"]:
        if beta_name in posterior:
            beta_arr = posterior[beta_name].values  # .values FIRST!
            beta_samples = _flatten_samples(beta_arr)

            # Try to get media data
            if hasattr(model, "panel") and model.panel is not None:
                try:
                    ch_idx = channels.index(channel)
                    X_media = model.panel.X_media
                    if hasattr(X_media, "values"):
                        X_media = X_media.values
                    media_sum = float(X_media[:, ch_idx].sum())
                    return beta_samples * media_sum * y_std
                except Exception:
                    pass

            return beta_samples * y_std

    return None


def compute_marginal_roi(
    model: Any,
    channel: str,
    spend_level: float | None = None,
    delta: float = 0.01,
    hdi_prob: float = 0.94,
) -> dict[str, float]:
    """
    Compute marginal ROI at a given spend level.

    Marginal ROI is the derivative of the response curve with respect to spend,
    measuring the incremental return from the next dollar invested.

    Parameters
    ----------
    model : BayesianMMM
        Fitted model
    channel : str
        Channel name
    spend_level : float, optional
        Spend level to evaluate. If None, uses current average spend.
    delta : float
        Relative change for numerical differentiation
    hdi_prob : float
        HDI probability

    Returns
    -------
    dict
        Marginal ROI statistics including mean, HDI, and comparison to average ROI
    """
    _check_model_fitted(model)

    posterior = _get_posterior(model)
    y_mean, y_std = _get_scaling_params(model)

    # Get current spend if not specified
    if spend_level is None:
        spend_data = _extract_spend_from_model(model)
        spend_level = spend_data.get(channel, 0.0)
        if spend_level <= 0:
            raise ValueError(f"No spend data for {channel}")
        # Use mean spend per period
        n_obs = getattr(model, "n_obs", 52)
        spend_level = spend_level / n_obs

    # Import saturation helpers (avoid circular import)
    from .saturation import _apply_saturation, _get_beta_samples, _get_saturation_params

    # Get saturation parameters
    sat_params = _get_saturation_params(model, posterior, channel)
    if sat_params is None:
        logger.warning(
            f"Cannot compute marginal ROI for {channel} - no saturation params"
        )
        return {"marginal_roi_mean": np.nan}

    # Get beta samples
    beta_samples = _get_beta_samples(posterior, channel)
    if beta_samples is None:
        return {"marginal_roi_mean": np.nan}

    # Compute marginal response via numerical differentiation
    spend_high = spend_level * (1 + delta)
    spend_low = spend_level * (1 - delta)

    # Apply saturation function
    response_high = _apply_saturation(spend_high, sat_params)
    response_low = _apply_saturation(spend_low, sat_params)

    # Marginal response = d(response)/d(spend)
    marginal_response = (response_high - response_low) / (spend_high - spend_low)

    # Marginal ROI = beta * marginal_response * y_std
    marginal_roi_samples = beta_samples * marginal_response * y_std

    mean_val = float(np.mean(marginal_roi_samples))
    lower, upper = _compute_hdi(marginal_roi_samples, hdi_prob)

    return {
        "marginal_roi_mean": mean_val,
        "marginal_roi_hdi_low": lower,
        "marginal_roi_hdi_high": upper,
        "spend_level": spend_level,
        "prob_marginal_positive": float(np.mean(marginal_roi_samples > 0)),
    }


__all__ = [
    "compute_roi_with_uncertainty",
    "compute_marginal_roi",
    "_extract_spend_from_model",
    "_get_contribution_samples",
]
