"""
ROI computation functions for MMM reporting.

Functions for computing ROI with uncertainty quantification from fitted models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ...model.component_scale import to_kpi_units
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
    from .measurement import (
        resolve_channel_divisor,
        spend_metric_meta,
    )

    _check_model_fitted(model)

    posterior = _get_posterior(model)
    channels = _get_channel_names(model)
    y_mean, y_std = _get_scaling_params(model)

    # An explicit external spend series is treated as monetary dollars (the
    # legacy override path). When absent, the measurement-aware resolver picks
    # the divisor AND the labels per channel (ROI for spend / cpm / cpc /
    # spend_column channels; efficiency per 1,000 impressions for cost-less
    # impression channels).
    external_spend: dict[str, float] | None = None
    if spend_data is not None:
        external_spend = (
            spend_data.to_dict() if isinstance(spend_data, pd.Series) else spend_data
        )

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

        # Resolve the divisor + metric metadata for this channel.
        if external_spend is not None:
            divisor = float(external_spend.get(channel, 0.0))
            meta = spend_metric_meta()
        else:
            resolved = resolve_channel_divisor(model, channel)
            divisor = resolved.total
            meta = resolved.meta
            if not resolved.found:
                logger.warning(f"No media data for {channel}, skipping ROI computation")
                continue

        if divisor <= 0:
            logger.warning(
                f"Non-positive divisor for {channel}, skipping ROI computation"
            )
            continue

        # Compute ROI / efficiency samples (incremental KPI per divisor unit).
        roi_samples = contrib_samples / divisor

        # Compute statistics
        contrib_mean = float(np.mean(contrib_samples))
        contrib_lower, contrib_upper = _compute_hdi(contrib_samples, hdi_prob)

        roi_mean = float(np.mean(roi_samples))
        roi_lower, roi_upper = _compute_hdi(roi_samples, hdi_prob)

        prob_positive = float(np.mean(roi_samples > 0))
        # P(profitable) only makes sense against a 1.0 break-even (a dollar
        # cost). For efficiency metrics there is no cost to beat — report None.
        prob_profitable = (
            float(np.mean(roi_samples > meta.reference))
            if meta.supports_profitability
            else None
        )

        results.append(
            ROIResult(
                channel=channel,
                spend=divisor,
                contribution_mean=contrib_mean,
                contribution_lower=contrib_lower,
                contribution_upper=contrib_upper,
                roi_mean=roi_mean,
                roi_lower=roi_lower,
                roi_upper=roi_upper,
                prob_positive=prob_positive,
                prob_profitable=prob_profitable,
                metric_is_monetary=meta.is_monetary,
                metric_label=meta.roi_label,
                marginal_label=meta.marginal_label,
                value_units=meta.value_units,
                divisor_units=meta.divisor_units,
                reference=meta.reference,
                measurement_unit=meta.unit.value,
                cost_basis=meta.cost_basis,
            )
        )

    return pd.DataFrame([r.to_dict() for r in results])


def _extract_spend_from_model(model: Any) -> dict[str, float]:
    """Per-channel ROI divisor (total) from the model.

    Thin wrapper over the measurement-aware resolver
    (:func:`mmm_framework.reporting.helpers.measurement.resolve_spend_dict`).
    For ordinary spend channels this is the summed media variable exactly as
    before; for impression/click channels it is the resolved divisor (a derived
    spend from cpm/cpc/spend_column, or the per-1,000-impression volume for the
    efficiency case).
    """
    from .measurement import resolve_spend_dict

    return resolve_spend_dict(model)


class ContributionWindowUnsupported(ValueError):
    """A windowed contribution was asked for on a shape that cannot carry one.

    Raised rather than returning the full-series value, which is
    self-consistent and wrong: a windowed request that silently answers the
    whole series is indistinguishable from a correct one at the call site.
    """


def _get_contribution_samples(
    model: Any,
    posterior: Any,
    channel: str,
    y_mean: float,
    y_std: float,
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract contribution samples for a channel.

    ``mask`` is an optional boolean array over observations. When given, the
    contribution is summed over the selected observations only. A per-draw
    SCALAR contribution has no observation axis to window, so a masked request
    against one raises :class:`ContributionWindowUnsupported` instead of
    quietly returning the full-series total.
    """
    if posterior is None:
        return None

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.all():
            mask = None  # full series: take the untouched, bit-stable path

    def _reduce_time(samples: np.ndarray, var_name: str) -> np.ndarray:
        """Sum a (samples, obs) array over the window, or over everything."""
        if samples.ndim <= 1:
            if mask is not None:
                raise ContributionWindowUnsupported(
                    f"{var_name!r} is a per-draw scalar for channel "
                    f"{channel!r}: it carries no observation axis, so it "
                    "cannot be restricted to a time window. Use an estimand "
                    "whose numerator is a counterfactual contrast, or drop the "
                    "window."
                )
            return samples
        if mask is not None:
            if samples.shape[-1] != mask.size:
                raise ContributionWindowUnsupported(
                    f"{var_name!r} has {samples.shape[-1]} values per draw for "
                    f"channel {channel!r} but the window mask covers "
                    f"{mask.size} observations, so the two cannot be aligned."
                )
            samples = samples[..., mask]
        return samples.sum(axis=-1)

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
            samples = _reduce_time(_flatten_samples(arr), var_name)
            return to_kpi_units(samples, model)

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
                    # The last axis is the channel axis only when its LENGTH
                    # says so. Assuming it unconditionally meant a
                    # (chain, draw, obs) variable — which has no channel axis —
                    # had its TIME axis indexed: `arr[..., ch_idx]` returned the
                    # value at period `ch_idx` instead of the window total, i.e.
                    # roughly 1/n_obs of the right answer, published as the
                    # channel's contribution. Falling through leaves a
                    # (samples, time) array for the sum below.
                    arr = da.values  # Get numpy FIRST
                    arr = _flatten_samples(arr)
                    if (
                        arr.ndim > 1
                        and arr.shape[-1] == len(channels)
                        and ch_idx < arr.shape[-1]
                    ):
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

            arr = _reduce_time(arr, "channel_contributions")

            # The scale of `channel_contributions` is a per-family convention,
            # not a constant: the core graph registers it standardized, the
            # extension graphs already multiplied by y_std. Hard-coding the
            # core rule published an extension model's contribution — and so
            # its ROI — inflated by a factor of y_std. Measured on a nested
            # model with y_std = 68.8: contribution 797,050 against a total KPI
            # of 88,685, i.e. nine times the whole KPI, rendered as ROI 132.9.
            return to_kpi_units(arr, model)

        except ContributionWindowUnsupported:
            # A refusal, not a failed extraction. Letting the broad handler
            # below swallow it would fall through to the beta fallback and
            # answer with a DIFFERENT number — which is the silent-wrong-answer
            # behaviour this refusal exists to prevent.
            raise
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
                    column = np.asarray(X_media)[:, ch_idx]
                    if mask is not None:
                        if column.size != mask.size:
                            raise ContributionWindowUnsupported(
                                f"media for {channel!r} has {column.size} rows "
                                f"but the window mask covers {mask.size}."
                            )
                        column = column[mask]
                    media_sum = float(column.sum())
                    return beta_samples * media_sum * y_std
                except Exception:
                    pass

            if mask is not None:
                raise ContributionWindowUnsupported(
                    f"only a coefficient is available for channel {channel!r} "
                    "and its media series could not be read, so there is no "
                    "observation axis to restrict to a window."
                )
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

    # Scale inputs by media_max to match how model was fitted
    if hasattr(model, "_media_max") and channel in model._media_max:
        scale_factor = model._media_max[channel] + 1e-8
    else:
        scale_factor = 1.0

    scaled_high = spend_high / scale_factor
    scaled_low = spend_low / scale_factor

    # Apply saturation function
    response_high = _apply_saturation(scaled_high, sat_params)
    response_low = _apply_saturation(scaled_low, sat_params)

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
    "ContributionWindowUnsupported",
]
