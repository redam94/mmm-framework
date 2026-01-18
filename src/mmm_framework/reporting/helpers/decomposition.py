"""
Component decomposition functions for MMM reporting.

Functions for computing model component decomposition with uncertainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .results import DecompositionResult
from .roi import _get_contribution_samples
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
    _get_scaling_params,
)


def compute_component_decomposition(
    model: Any,
    include_time_series: bool = True,
    hdi_prob: float = 0.94,
) -> list[DecompositionResult]:
    """
    Compute full component decomposition of model predictions.

    Breaks down total outcome into contributions from:
    - Baseline/intercept
    - Trend
    - Seasonality
    - Media channels (individually)
    - Control variables
    - Geographic/product effects (if applicable)

    Parameters
    ----------
    model : BayesianMMM
        Fitted model
    include_time_series : bool
        Whether to include time series arrays
    hdi_prob : float
        HDI probability

    Returns
    -------
    list[DecompositionResult]
        Decomposition results by component

    Examples
    --------
    >>> decomp = compute_component_decomposition(mmm)
    >>> df = pd.DataFrame([d.to_dict() for d in decomp])
    >>> print(df[['component', 'total_contribution', 'pct_of_total']])
    """
    _check_model_fitted(model)

    # Try model's built-in method first
    if hasattr(model, "compute_component_decomposition"):
        try:
            decomp = model.compute_component_decomposition()
            return _convert_model_decomposition(decomp, hdi_prob)
        except Exception as e:
            logger.warning(f"Model decomposition failed: {e}")

    # Manual computation from trace
    return _compute_decomposition_from_trace(model, include_time_series, hdi_prob)


def _convert_model_decomposition(
    decomp: Any,
    hdi_prob: float,
) -> list[DecompositionResult]:
    """Convert model's ComponentDecomposition to DecompositionResult list."""
    results = []

    # Calculate total for percentages
    total = (
        abs(decomp.total_intercept)
        + abs(decomp.total_trend)
        + abs(decomp.total_seasonality)
        + abs(decomp.total_media)
        + abs(decomp.total_controls)
    )

    if total == 0:
        total = 1.0

    # Baseline
    results.append(
        DecompositionResult(
            component="Baseline",
            total_contribution=decomp.total_intercept,
            contribution_lower=decomp.total_intercept,  # No uncertainty from built-in
            contribution_upper=decomp.total_intercept,
            pct_of_total=decomp.total_intercept / total,
            time_series=decomp.intercept,
        )
    )

    # Trend
    if decomp.total_trend != 0:
        results.append(
            DecompositionResult(
                component="Trend",
                total_contribution=decomp.total_trend,
                contribution_lower=decomp.total_trend,
                contribution_upper=decomp.total_trend,
                pct_of_total=decomp.total_trend / total,
                time_series=decomp.trend,
            )
        )

    # Seasonality
    if decomp.total_seasonality != 0:
        results.append(
            DecompositionResult(
                component="Seasonality",
                total_contribution=decomp.total_seasonality,
                contribution_lower=decomp.total_seasonality,
                contribution_upper=decomp.total_seasonality,
                pct_of_total=decomp.total_seasonality / total,
                time_series=decomp.seasonality,
            )
        )

    # Media channels
    if decomp.media_by_channel is not None:
        for ch in decomp.media_by_channel.columns:
            ch_total = float(decomp.media_by_channel[ch].sum())
            results.append(
                DecompositionResult(
                    component=ch,
                    total_contribution=ch_total,
                    contribution_lower=ch_total,
                    contribution_upper=ch_total,
                    pct_of_total=ch_total / total,
                    time_series=decomp.media_by_channel[ch].values,
                )
            )

    # Controls
    if decomp.total_controls != 0:
        results.append(
            DecompositionResult(
                component="Controls",
                total_contribution=decomp.total_controls,
                contribution_lower=decomp.total_controls,
                contribution_upper=decomp.total_controls,
                pct_of_total=decomp.total_controls / total,
                time_series=decomp.controls_total,
            )
        )

    return results


def _compute_decomposition_from_trace(
    model: Any,
    include_time_series: bool,
    hdi_prob: float,
) -> list[DecompositionResult]:
    """Compute decomposition directly from trace."""
    posterior = _get_posterior(model)
    y_mean, y_std = _get_scaling_params(model)
    n_obs = getattr(model, "n_obs", 52)

    results = []
    total = 0.0

    # Intercept
    if "intercept" in posterior:
        intercept_samples = _flatten_samples(posterior["intercept"].values)
        intercept_mean = float(np.mean(intercept_samples)) * y_std
        intercept_lower, intercept_upper = _compute_hdi(
            intercept_samples * y_std, hdi_prob
        )
        total_intercept = intercept_mean * n_obs
        total += abs(total_intercept)

        results.append(
            DecompositionResult(
                component="Baseline",
                total_contribution=total_intercept,
                contribution_lower=intercept_lower * n_obs,
                contribution_upper=intercept_upper * n_obs,
                pct_of_total=0.0,  # Will update after total calculated
                time_series=(
                    np.full(n_obs, intercept_mean) if include_time_series else None
                ),
            )
        )

    # Media channels
    channels = _get_channel_names(model)
    for channel in channels:
        contrib_samples = _get_contribution_samples(
            model, posterior, channel, y_mean, y_std
        )
        if contrib_samples is not None:
            contrib_mean = float(np.mean(contrib_samples))
            contrib_lower, contrib_upper = _compute_hdi(contrib_samples, hdi_prob)
            total += abs(contrib_mean)

            results.append(
                DecompositionResult(
                    component=channel,
                    total_contribution=contrib_mean,
                    contribution_lower=contrib_lower,
                    contribution_upper=contrib_upper,
                    pct_of_total=0.0,
                )
            )

    # Update percentages
    if total > 0:
        for r in results:
            r.pct_of_total = r.total_contribution / total

    return results


def compute_decomposition_waterfall(
    decomp: list[DecompositionResult],
    start_label: str = "Starting Value",
    end_label: str = "Total Outcome",
) -> pd.DataFrame:
    """
    Format decomposition for waterfall chart visualization.

    Parameters
    ----------
    decomp : list[DecompositionResult]
        Decomposition results
    start_label : str
        Label for starting point
    end_label : str
        Label for ending total

    Returns
    -------
    pd.DataFrame
        DataFrame formatted for waterfall chart
    """
    rows = []
    running_total = 0.0

    for d in decomp:
        rows.append(
            {
                "component": d.component,
                "contribution": d.total_contribution,
                "start": running_total,
                "end": running_total + d.total_contribution,
                "pct": d.pct_of_total,
            }
        )
        running_total += d.total_contribution

    # Add total row
    rows.append(
        {
            "component": end_label,
            "contribution": running_total,
            "start": 0,
            "end": running_total,
            "pct": 1.0,
        }
    )

    return pd.DataFrame(rows)


__all__ = [
    "compute_component_decomposition",
    "compute_decomposition_waterfall",
    "_convert_model_decomposition",
    "_compute_decomposition_from_trace",
]
