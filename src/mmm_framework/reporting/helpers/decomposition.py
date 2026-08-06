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


#: Below this ratio of |signed total| to sum-of-magnitudes, the signed
#: denominator is numerically unusable and percentages are suppressed. Measured
#: on a near-zero signed total, a naive signed share renders Baseline -1105.8%
#: and Trend +1613.0% — arithmetically correct and useless.
_SIGNED_SHARE_MIN_RATIO = 0.10


def _share_denominator(values: "list[float]") -> tuple[float, bool]:
    """(denominator, shares_are_meaningful) for component percentages.

    The denominator must be the SIGNED total: components sum to the fitted
    outcome, and a sum of magnitudes is not that sum whenever any component is
    negative (a declining trend, a negative control), so shares computed
    against it do not add to 1.

    Falls back to the magnitude sum, flagged, when the signed total is so close
    to zero that shares explode — better an admittedly-approximate denominator
    than a page of ±1000% figures.
    """
    signed = float(sum(values))
    magnitude = float(sum(abs(v) for v in values))
    if magnitude <= 0:
        return 1.0, False
    if abs(signed) < _SIGNED_SHARE_MIN_RATIO * magnitude:
        return magnitude, False
    return signed, True


#: Optional component blocks, as ``(row label, total field, series field)``.
#: Present on ``ComponentDecomposition`` only when the corresponding block was
#: fit, so each is emitted on a ``is not None`` test rather than a ``!= 0`` one:
#: a fitted block whose total happens to net to zero is still part of the
#: identity, and dropping it would move the share denominator.
#:
#: The labels match ``extractors/bayesian.py`` exactly. They have to: both paths
#: describe the same decomposition, and this one used to omit all five, so a
#: model with events or levers had shares that did not add to 1 and rows that
#: appeared or vanished depending on which path ran.
_OPTIONAL_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("Events", "total_events", "events"),
    ("Synergy", "total_interactions", "interactions"),
    ("Price & Promotion", "total_levers", "levers"),
    ("Geo", "total_geo", "geo_effects"),
    ("Product", "total_product", "product_effects"),
)


def _optional_totals(decomp: Any) -> list[tuple[str, float, Any]]:
    """``(label, total, series)`` for each optional block actually fit."""
    out: list[tuple[str, float, Any]] = []
    for label, total_field, series_field in _OPTIONAL_COMPONENTS:
        total = getattr(decomp, total_field, None)
        if total is None:
            continue
        value = float(total)
        if not np.isfinite(value):
            continue
        out.append((label, value, getattr(decomp, series_field, None)))
    return out


def _convert_model_decomposition(
    decomp: Any,
    hdi_prob: float,
) -> list[DecompositionResult]:
    """Convert model's ComponentDecomposition to DecompositionResult list."""
    results = []

    optional = _optional_totals(decomp)

    # Calculate total for percentages. SIGNED, not a sum of magnitudes: the
    # components sum to the fitted outcome, so a magnitude denominator makes
    # shares fail to add to 1 the moment any component is negative.
    #
    # Every fitted block belongs in this denominator, including the optional
    # ones. Omitting them made the shares of the remaining rows too large by
    # exactly the omitted blocks' share.
    total, _shares_ok = _share_denominator(
        [
            decomp.total_intercept,
            decomp.total_trend,
            decomp.total_seasonality,
            decomp.total_media,
            decomp.total_controls,
        ]
        + [value for _label, value, _series in optional]
    )

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

    # Events, synergy, levers, geo, product. These were previously dropped from
    # both the rows and the denominator, so a model that fit any of them
    # reported a decomposition that did not describe it.
    for label, value, series in optional:
        results.append(
            DecompositionResult(
                component=label,
                total_contribution=value,
                contribution_lower=value,
                contribution_upper=value,
                pct_of_total=value / total,
                time_series=series,
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
    # Signed component totals, accumulated for the share denominator. Must be
    # initialised here, not inside the intercept branch: a model without an
    # `intercept` posterior var would otherwise hit an unbound name.
    component_totals: list[float] = []

    # Intercept
    if "intercept" in posterior:
        intercept_samples = _flatten_samples(posterior["intercept"].values)
        # `+ y_mean` is required: the standardized intercept unstandardizes as
        # `intercept * y_std + y_mean`. Omitting it left this fallback Baseline
        # short by `y_mean * n_obs` — and this path is taken precisely when
        # compute_component_decomposition() raised, i.e. when a trustworthy
        # number matters most.
        intercept_mean = float(np.mean(intercept_samples)) * y_std + y_mean
        intercept_lower, intercept_upper = _compute_hdi(
            intercept_samples * y_std + y_mean, hdi_prob
        )
        total_intercept = intercept_mean * n_obs
        component_totals.append(total_intercept)

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
            component_totals.append(contrib_mean)

            results.append(
                DecompositionResult(
                    component=channel,
                    total_contribution=contrib_mean,
                    contribution_lower=contrib_lower,
                    contribution_upper=contrib_upper,
                    pct_of_total=0.0,
                )
            )

    # Update percentages against the SIGNED total (see _share_denominator).
    total, _shares_ok = _share_denominator(component_totals)
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
