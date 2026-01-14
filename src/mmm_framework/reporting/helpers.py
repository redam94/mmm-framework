"""
Reporting Helper Functions for MMM Framework.

This module provides comprehensive helper functions for computing and visualizing
key MMM outputs with proper uncertainty quantification:

- ROI computation with credible intervals
- Prior vs posterior comparison plots
- Adstock and saturation effect visualization
- Component decomposition analysis
- Extended model support (NestedMMM, MultivariateMMM, CombinedMMM)

All functions are designed to work with both BayesianMMM and extended model classes,
extracting data from traces and computing uncertainty-aware metrics.

Usage:
    from mmm_framework.reporting.helpers import (
        compute_roi_with_uncertainty,
        compute_channel_contributions,
        get_prior_posterior_comparison,
        compute_saturation_curves_with_uncertainty,
        compute_adstock_weights,
        compute_component_decomposition,
    )
    
    # After fitting a model
    roi_df = compute_roi_with_uncertainty(mmm, spend_data)
    prior_post = get_prior_posterior_comparison(mmm)
    sat_curves = compute_saturation_curves_with_uncertainty(mmm)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import numpy as np
import pandas as pd
from loguru import logger

try:
    import arviz as az
except ImportError:
    az = None

try:
    import pymc as pm
except ImportError:
    pm = None


# =============================================================================
# Type Definitions and Protocols
# =============================================================================


@runtime_checkable
class HasTrace(Protocol):
    """Protocol for objects with ArviZ trace."""
    @property
    def _trace(self) -> Any: ...


@runtime_checkable
class HasModel(Protocol):
    """Protocol for objects with PyMC model."""
    @property
    def model(self) -> Any: ...
    @property
    def _model(self) -> Any: ...


@runtime_checkable
class HasPanel(Protocol):
    """Protocol for objects with panel data."""
    @property
    def panel(self) -> Any: ...


# =============================================================================
# Result Containers
# =============================================================================


@dataclass
class ROIResult:
    """Container for ROI computation results with uncertainty."""
    
    channel: str
    spend: float
    contribution_mean: float
    contribution_lower: float
    contribution_upper: float
    roi_mean: float
    roi_lower: float
    roi_upper: float
    prob_positive: float  # P(ROI > 0)
    prob_profitable: float  # P(ROI > 1)
    marginal_roi_mean: float | None = None
    marginal_roi_lower: float | None = None
    marginal_roi_upper: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "spend": self.spend,
            "contribution_mean": self.contribution_mean,
            "contribution_hdi_low": self.contribution_lower,
            "contribution_hdi_high": self.contribution_upper,
            "roi_mean": self.roi_mean,
            "roi_hdi_low": self.roi_lower,
            "roi_hdi_high": self.roi_upper,
            "prob_positive": self.prob_positive,
            "prob_profitable": self.prob_profitable,
            "marginal_roi_mean": self.marginal_roi_mean,
            "marginal_roi_hdi_low": self.marginal_roi_lower,
            "marginal_roi_hdi_high": self.marginal_roi_upper,
        }


@dataclass
class PriorPosteriorComparison:
    """Container for prior vs posterior comparison."""
    
    parameter: str
    prior_mean: float | None
    prior_sd: float | None
    posterior_mean: float
    posterior_sd: float
    posterior_hdi_low: float
    posterior_hdi_high: float
    shrinkage: float | None  # 1 - (posterior_sd / prior_sd)
    prior_samples: np.ndarray | None
    posterior_samples: np.ndarray
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "prior_mean": self.prior_mean,
            "prior_sd": self.prior_sd,
            "posterior_mean": self.posterior_mean,
            "posterior_sd": self.posterior_sd,
            "posterior_hdi_low": self.posterior_hdi_low,
            "posterior_hdi_high": self.posterior_hdi_high,
            "shrinkage": self.shrinkage,
        }


@dataclass
class SaturationCurveResult:
    """Container for saturation curve with uncertainty."""
    
    channel: str
    spend_grid: np.ndarray
    response_mean: np.ndarray
    response_lower: np.ndarray
    response_upper: np.ndarray
    current_spend: float
    current_response: float
    saturation_level: float  # % of max response at current spend
    marginal_response_at_current: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "spend": self.spend_grid.tolist(),
            "response_mean": self.response_mean.tolist(),
            "response_hdi_low": self.response_lower.tolist(),
            "response_hdi_high": self.response_upper.tolist(),
            "current_spend": self.current_spend,
            "current_response": self.current_response,
            "saturation_level": self.saturation_level,
            "marginal_response_at_current": self.marginal_response_at_current,
        }


@dataclass
class AdstockResult:
    """Container for adstock decay curve."""
    
    channel: str
    decay_weights: np.ndarray
    alpha_mean: float
    alpha_lower: float
    alpha_upper: float
    half_life: float  # Periods until 50% decay
    total_carryover: float  # Sum of weights beyond t=0
    l_max: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "decay_weights": self.decay_weights.tolist(),
            "alpha_mean": self.alpha_mean,
            "alpha_hdi_low": self.alpha_lower,
            "alpha_hdi_high": self.alpha_upper,
            "half_life": self.half_life,
            "total_carryover": self.total_carryover,
            "l_max": self.l_max,
        }


@dataclass
class DecompositionResult:
    """Container for model component decomposition."""
    
    component: str
    total_contribution: float
    contribution_lower: float
    contribution_upper: float
    pct_of_total: float
    time_series: np.ndarray | None = None
    time_series_lower: np.ndarray | None = None
    time_series_upper: np.ndarray | None = None
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "component": self.component,
            "total_contribution": self.total_contribution,
            "contribution_hdi_low": self.contribution_lower,
            "contribution_hdi_high": self.contribution_upper,
            "pct_of_total": self.pct_of_total,
        }
        if self.time_series is not None:
            result["time_series"] = self.time_series.tolist()
        return result


@dataclass 
class MediatedEffectResult:
    """Container for mediated (indirect) effect decomposition."""
    
    channel: str
    outcome: str
    direct_mean: float
    direct_lower: float
    direct_upper: float
    indirect_mean: float
    indirect_lower: float
    indirect_upper: float
    total_mean: float
    total_lower: float
    total_upper: float
    proportion_mediated: float
    mediator_breakdown: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "outcome": self.outcome,
            "direct_mean": self.direct_mean,
            "direct_hdi_low": self.direct_lower,
            "direct_hdi_high": self.direct_upper,
            "indirect_mean": self.indirect_mean,
            "indirect_hdi_low": self.indirect_lower,
            "indirect_hdi_high": self.indirect_upper,
            "total_mean": self.total_mean,
            "total_hdi_low": self.total_lower,
            "total_hdi_high": self.total_upper,
            "proportion_mediated": self.proportion_mediated,
            "mediator_breakdown": self.mediator_breakdown,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def _compute_hdi(
    samples: np.ndarray,
    prob: float = 0.94,
) -> tuple[float, float]:
    """
    Compute highest density interval from samples.
    
    Parameters
    ----------
    samples : np.ndarray
        Posterior samples (1D array)
    prob : float
        Probability mass for HDI (default 0.94)
    
    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of HDI
    """
    samples = np.asarray(samples).flatten()
    samples = samples[~np.isnan(samples)]
    
    if len(samples) == 0:
        return np.nan, np.nan
    
    if az is not None:
        try:
            hdi = az.hdi(samples, hdi_prob=prob)
            return float(hdi[0]), float(hdi[1])
        except Exception:
            pass
    
    # Fallback to percentile-based interval
    alpha = (1 - prob) / 2
    return float(np.percentile(samples, alpha * 100)), float(np.percentile(samples, (1 - alpha) * 100))


def _get_trace(model: Any) -> Any | None:
    """Extract ArviZ trace from model."""
    if hasattr(model, "_trace"):
        return model._trace
    if hasattr(model, "trace"):
        return model.trace
    return None


def _get_posterior(model: Any) -> Any | None:
    """Extract posterior from trace."""
    trace = _get_trace(model)
    if trace is not None and hasattr(trace, "posterior"):
        return trace.posterior
    return None


def _get_channel_names(model: Any) -> list[str]:
    """Extract channel names from model."""
    if hasattr(model, "channel_names"):
        return list(model.channel_names)
    if hasattr(model, "panel") and model.panel is not None:
        if hasattr(model.panel, "channel_names"):
            return list(model.panel.channel_names)
    return []


def _get_scaling_params(model: Any) -> tuple[float, float]:
    """Get y_mean and y_std from model for rescaling."""
    y_mean = getattr(model, "y_mean", 0.0)
    y_std = getattr(model, "y_std", 1.0)
    return float(y_mean), float(y_std)


def _flatten_samples(arr: np.ndarray) -> np.ndarray:
    """Flatten chain and draw dimensions from posterior samples."""
    if arr.ndim >= 2:
        return arr.reshape(-1, *arr.shape[2:]) if arr.ndim > 2 else arr.flatten()
    return arr.flatten()


def _check_model_fitted(model: Any) -> None:
    """Raise error if model is not fitted."""
    trace = _get_trace(model)
    if trace is None:
        raise ValueError("Model not fitted. Call fit() first.")


# =============================================================================
# ROI Computation
# =============================================================================


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
        
        results.append(ROIResult(
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
        ))
    
    return pd.DataFrame([r.to_dict() for r in results])


def _extract_spend_from_model(model: Any) -> dict[str, float]:
    """Extract total spend per channel from model's panel data."""
    spend = {}
    channels = _get_channel_names(model)
    
    # Try panel data
    if hasattr(model, "panel") and model.panel is not None:
        panel = model.panel
        if hasattr(panel, "X_media") and panel.X_media is not None:
            X_media = panel.X_media
            for i, ch in enumerate(channels):
                if i < X_media.shape[1]:
                    spend[ch] = float(X_media[:, i].sum())
    
    # Try X_media_raw
    if not spend and hasattr(model, "X_media_raw"):
        X_media = model.X_media_raw
        for i, ch in enumerate(channels):
            if i < X_media.shape[1]:
                spend[ch] = float(X_media[:, i].sum())
    
    # Try X_media
    if not spend and hasattr(model, "X_media"):
        X_media = model.X_media
        for i, ch in enumerate(channels):
            if i < X_media.shape[1]:
                spend[ch] = float(X_media[:, i].sum())
    
    return spend


def _get_contribution_samples(
    model: Any,
    posterior: Any,
    channel: str,
    y_mean: float,
    y_std: float,
) -> np.ndarray | None:
    """
    Extract contribution samples for a channel.
    
    Tries multiple variable naming conventions used across model types.
    """
    if posterior is None:
        return None
    
    # Try different variable naming conventions
    possible_names = [
        f"contribution_{channel}",
        f"channel_contribution_{channel}",
        f"media_contribution_{channel}",
    ]
    
    for var_name in possible_names:
        if var_name in posterior:
            samples = posterior[var_name].values
            samples = _flatten_samples(samples)
            # Sum over time if needed
            if samples.ndim > 1:
                samples = samples.sum(axis=-1)
            # Scale to original units
            return samples * y_std
    
    # Fall back to computing from beta and transformed media
    beta_names = [f"beta_{channel}", f"beta_media_{channel}"]
    for beta_name in beta_names:
        if beta_name in posterior:
            beta_samples = _flatten_samples(posterior[beta_name].values)
            
            # Try to get channel contributions deterministic
            if "channel_contributions" in posterior:
                ch_idx = _get_channel_names(model).index(channel)
                contrib = posterior["channel_contributions"].values
                contrib = _flatten_samples(contrib)
                if contrib.ndim > 1:
                    contrib = contrib[:, :, ch_idx].sum(axis=-1)
                return contrib * y_std
            
            # Rough estimate from beta * mean_media
            if hasattr(model, "panel") and model.panel is not None:
                ch_idx = _get_channel_names(model).index(channel)
                X_media = model.panel.X_media
                if X_media is not None:
                    media_sum = X_media[:, ch_idx].sum()
                    return beta_samples * media_sum * y_std
            
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
    
    # Get saturation parameters
    sat_params = _get_saturation_params(model, posterior, channel)
    if sat_params is None:
        logger.warning(f"Cannot compute marginal ROI for {channel} - no saturation params")
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


# =============================================================================
# Prior vs Posterior Comparison
# =============================================================================


def get_prior_posterior_comparison(
    model: Any,
    parameters: list[str] | None = None,
    n_prior_samples: int = 1000,
    hdi_prob: float = 0.94,
    random_seed: int = 42,
) -> list[PriorPosteriorComparison]:
    """
    Compute prior vs posterior comparison for model parameters.
    
    Shows how the data updated prior beliefs, computing shrinkage metrics
    that quantify how informative the data was for each parameter.
    
    Parameters
    ----------
    model : BayesianMMM or ExtendedMMM
        Fitted model with trace
    parameters : list[str], optional
        Parameters to compare. If None, auto-selects key parameters.
    n_prior_samples : int
        Number of prior predictive samples to draw
    hdi_prob : float
        HDI probability for posterior intervals
    random_seed : int
        Random seed for prior sampling
    
    Returns
    -------
    list[PriorPosteriorComparison]
        Comparison results for each parameter
    
    Examples
    --------
    >>> comparisons = get_prior_posterior_comparison(mmm)
    >>> for c in comparisons:
    ...     print(f"{c.parameter}: shrinkage = {c.shrinkage:.2%}")
    """
    _check_model_fitted(model)
    
    posterior = _get_posterior(model)
    pymc_model = getattr(model, "_model", None) or getattr(model, "model", None)
    
    if posterior is None:
        raise ValueError("No posterior found in model trace")
    
    # Auto-select parameters if not specified
    if parameters is None:
        parameters = _select_key_parameters(posterior, model)
    
    # Sample from prior
    prior_samples = {}
    if pymc_model is not None and pm is not None:
        try:
            with pymc_model:
                prior = pm.sample_prior_predictive(
                    samples=n_prior_samples,
                    random_seed=random_seed,
                )
            if hasattr(prior, "prior"):
                prior_samples = {
                    var: prior.prior[var].values.flatten()
                    for var in prior.prior.data_vars
                }
        except Exception as e:
            logger.warning(f"Could not sample prior: {e}")
    
    results = []
    
    for param in parameters:
        if param not in posterior:
            continue
        
        # Get posterior samples
        post_vals = posterior[param].values
        post_samples = _flatten_samples(post_vals)
        
        # Handle multi-dimensional parameters
        if post_samples.ndim > 1:
            post_samples = post_samples.flatten()
        
        post_mean = float(np.mean(post_samples))
        post_sd = float(np.std(post_samples))
        post_lower, post_upper = _compute_hdi(post_samples, hdi_prob)
        
        # Get prior samples
        prior_vals = prior_samples.get(param)
        prior_mean = None
        prior_sd = None
        shrinkage = None
        
        if prior_vals is not None and len(prior_vals) > 0:
            prior_mean = float(np.mean(prior_vals))
            prior_sd = float(np.std(prior_vals))
            if prior_sd > 0:
                shrinkage = 1 - (post_sd / prior_sd)
        
        results.append(PriorPosteriorComparison(
            parameter=param,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            posterior_mean=post_mean,
            posterior_sd=post_sd,
            posterior_hdi_low=post_lower,
            posterior_hdi_high=post_upper,
            shrinkage=shrinkage,
            prior_samples=prior_vals,
            posterior_samples=post_samples,
        ))
    
    return results


def _select_key_parameters(posterior: Any, model: Any) -> list[str]:
    """Auto-select key parameters for prior-posterior comparison."""
    params = []
    channels = _get_channel_names(model)
    
    # Beta (media coefficients)
    for ch in channels:
        for prefix in ["beta_", "beta_media_"]:
            name = f"{prefix}{ch}"
            if name in posterior:
                params.append(name)
                break
    
    # Adstock parameters
    for ch in channels:
        for prefix in ["adstock_", "alpha_"]:
            name = f"{prefix}{ch}"
            if name in posterior:
                params.append(name)
                break
    
    # Saturation parameters
    for ch in channels:
        for prefix in ["sat_lam_", "saturation_", "kappa_", "slope_"]:
            name = f"{prefix}{ch}"
            if name in posterior:
                params.append(name)
                break
    
    # Global parameters
    for name in ["sigma", "intercept", "trend_slope"]:
        if name in posterior:
            params.append(name)
    
    return params


def compute_shrinkage_summary(
    comparisons: list[PriorPosteriorComparison],
) -> pd.DataFrame:
    """
    Summarize shrinkage across parameters.
    
    Parameters
    ----------
    comparisons : list[PriorPosteriorComparison]
        Results from get_prior_posterior_comparison
    
    Returns
    -------
    pd.DataFrame
        Summary with shrinkage metrics and data informativeness
    """
    rows = []
    for c in comparisons:
        rows.append({
            "parameter": c.parameter,
            "prior_mean": c.prior_mean,
            "prior_sd": c.prior_sd,
            "posterior_mean": c.posterior_mean,
            "posterior_sd": c.posterior_sd,
            "shrinkage": c.shrinkage,
            "data_informative": "Yes" if c.shrinkage and c.shrinkage > 0.5 else "No" if c.shrinkage else "Unknown",
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# Saturation Curves
# =============================================================================


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
            sat_params = {k: v[idx] if isinstance(v, np.ndarray) else v 
                         for k, v in sat_params.items()}
        
        # Compute response curves
        response_samples = np.zeros((len(beta_samples), n_points))
        for i in range(n_points):
            saturated = _apply_saturation(spend_grid[i], sat_params)
            response_samples[:, i] = beta_samples * saturated * y_std
        
        # Compute statistics
        response_mean = response_samples.mean(axis=0)
        response_lower = np.percentile(response_samples, (1 - hdi_prob) / 2 * 100, axis=0)
        response_upper = np.percentile(response_samples, (1 + hdi_prob) / 2 * 100, axis=0)
        
        # Current spend position
        current_spend = total_spend / n_obs
        current_idx = np.argmin(np.abs(spend_grid - current_spend))
        current_response = float(response_mean[current_idx])
        
        # Saturation level (% of max response)
        max_response = float(response_mean[-1]) if response_mean[-1] > 0 else 1.0
        saturation_level = current_response / max_response if max_response > 0 else 0.0
        
        # Marginal response at current spend
        if current_idx > 0 and current_idx < n_points - 1:
            marginal = (response_mean[current_idx + 1] - response_mean[current_idx - 1]) / (
                spend_grid[current_idx + 1] - spend_grid[current_idx - 1]
            )
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
        return x ** slope / (kappa ** slope + x ** slope)
    
    elif sat_type == "logistic":
        lam = params["lam"]
        return 1 / (1 + np.exp(-lam * (x - 0.5)))
    
    else:
        # Linear (no saturation)
        return np.ones_like(params.get("lam", np.array([1.0]))) * x


# =============================================================================
# Adstock Effects
# =============================================================================


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
        weights = alpha_mean ** lags
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


# =============================================================================
# Component Decomposition
# =============================================================================


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
    total = abs(decomp.total_intercept) + abs(decomp.total_trend) + \
            abs(decomp.total_seasonality) + abs(decomp.total_media) + \
            abs(decomp.total_controls)
    
    if total == 0:
        total = 1.0
    
    # Baseline
    results.append(DecompositionResult(
        component="Baseline",
        total_contribution=decomp.total_intercept,
        contribution_lower=decomp.total_intercept,  # No uncertainty from built-in
        contribution_upper=decomp.total_intercept,
        pct_of_total=decomp.total_intercept / total,
        time_series=decomp.intercept,
    ))
    
    # Trend
    if decomp.total_trend != 0:
        results.append(DecompositionResult(
            component="Trend",
            total_contribution=decomp.total_trend,
            contribution_lower=decomp.total_trend,
            contribution_upper=decomp.total_trend,
            pct_of_total=decomp.total_trend / total,
            time_series=decomp.trend,
        ))
    
    # Seasonality
    if decomp.total_seasonality != 0:
        results.append(DecompositionResult(
            component="Seasonality",
            total_contribution=decomp.total_seasonality,
            contribution_lower=decomp.total_seasonality,
            contribution_upper=decomp.total_seasonality,
            pct_of_total=decomp.total_seasonality / total,
            time_series=decomp.seasonality,
        ))
    
    # Media channels
    if decomp.media_by_channel is not None:
        for ch in decomp.media_by_channel.columns:
            ch_total = float(decomp.media_by_channel[ch].sum())
            results.append(DecompositionResult(
                component=ch,
                total_contribution=ch_total,
                contribution_lower=ch_total,
                contribution_upper=ch_total,
                pct_of_total=ch_total / total,
                time_series=decomp.media_by_channel[ch].values,
            ))
    
    # Controls
    if decomp.total_controls != 0:
        results.append(DecompositionResult(
            component="Controls",
            total_contribution=decomp.total_controls,
            contribution_lower=decomp.total_controls,
            contribution_upper=decomp.total_controls,
            pct_of_total=decomp.total_controls / total,
            time_series=decomp.controls_total,
        ))
    
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
        intercept_lower, intercept_upper = _compute_hdi(intercept_samples * y_std, hdi_prob)
        total_intercept = intercept_mean * n_obs
        total += abs(total_intercept)
        
        results.append(DecompositionResult(
            component="Baseline",
            total_contribution=total_intercept,
            contribution_lower=intercept_lower * n_obs,
            contribution_upper=intercept_upper * n_obs,
            pct_of_total=0.0,  # Will update after total calculated
            time_series=np.full(n_obs, intercept_mean) if include_time_series else None,
        ))
    
    # Media channels
    channels = _get_channel_names(model)
    for channel in channels:
        contrib_samples = _get_contribution_samples(model, posterior, channel, y_mean, y_std)
        if contrib_samples is not None:
            contrib_mean = float(np.mean(contrib_samples))
            contrib_lower, contrib_upper = _compute_hdi(contrib_samples, hdi_prob)
            total += abs(contrib_mean)
            
            results.append(DecompositionResult(
                component=channel,
                total_contribution=contrib_mean,
                contribution_lower=contrib_lower,
                contribution_upper=contrib_upper,
                pct_of_total=0.0,
            ))
    
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
        rows.append({
            "component": d.component,
            "contribution": d.total_contribution,
            "start": running_total,
            "end": running_total + d.total_contribution,
            "pct": d.pct_of_total,
        })
        running_total += d.total_contribution
    
    # Add total row
    rows.append({
        "component": end_label,
        "contribution": running_total,
        "start": 0,
        "end": running_total,
        "pct": 1.0,
    })
    
    return pd.DataFrame(rows)


# =============================================================================
# Extended Model Helpers
# =============================================================================


def compute_mediated_effects(
    model: Any,
    hdi_prob: float = 0.94,
) -> list[MediatedEffectResult]:
    """
    Compute direct, indirect, and total effects for nested/combined models.
    
    For models with mediators (e.g., Media → Awareness → Sales), this
    decomposes effects into direct and indirect pathways.
    
    Parameters
    ----------
    model : NestedMMM or CombinedMMM
        Fitted nested or combined model
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    list[MediatedEffectResult]
        Effect decomposition by channel and outcome
    
    Examples
    --------
    >>> effects = compute_mediated_effects(nested_mmm)
    >>> for e in effects:
    ...     print(f"{e.channel}: {e.proportion_mediated:.0%} mediated")
    """
    _check_model_fitted(model)
    
    # Check if model has mediation
    if not hasattr(model, "get_mediation_effects") and not hasattr(model, "mediator_names"):
        raise ValueError("Model does not support mediation analysis")
    
    # Try model's built-in method (but verify it returns valid data)
    if hasattr(model, "get_mediation_effects"):
        try:
            df = model.get_mediation_effects()
            # Verify it's actually a DataFrame with expected columns
            if isinstance(df, pd.DataFrame) and len(df) > 0 and "channel" in df.columns:
                return _convert_mediation_df(df, hdi_prob)
        except Exception as e:
            logger.debug(f"Model mediation method failed or returned invalid data: {e}")
    
    # Manual computation from trace
    return _compute_mediation_from_trace(model, hdi_prob)
    
    # Manual computation
    return _compute_mediation_from_trace(model, hdi_prob)


def _convert_mediation_df(df: pd.DataFrame, hdi_prob: float) -> list[MediatedEffectResult]:
    """Convert model's mediation DataFrame to MediatedEffectResult list."""
    results = []
    
    for _, row in df.iterrows():
        total = row.get("total_effect", 0)
        direct = row.get("direct_effect", 0)
        indirect = row.get("total_indirect", total - direct)
        
        prop_mediated = indirect / total if total != 0 else 0.0
        
        results.append(MediatedEffectResult(
            channel=row.get("channel", ""),
            outcome=row.get("outcome", "sales"),
            direct_mean=direct,
            direct_lower=direct - 1.96 * row.get("direct_effect_sd", 0),
            direct_upper=direct + 1.96 * row.get("direct_effect_sd", 0),
            indirect_mean=indirect,
            indirect_lower=indirect,  # Would need samples for proper HDI
            indirect_upper=indirect,
            total_mean=total,
            total_lower=total,
            total_upper=total,
            proportion_mediated=prop_mediated,
            mediator_breakdown=row.get("indirect_effects", {}),
        ))
    
    return results


def _compute_mediation_from_trace(
    model: Any,
    hdi_prob: float,
) -> list[MediatedEffectResult]:
    """Compute mediation effects from trace."""
    posterior = _get_posterior(model)
    channels = _get_channel_names(model)
    
    if posterior is None:
        logger.warning("No posterior found for mediation computation")
        return []
    
    results = []
    
    # Get outcome names
    outcome_names = getattr(model, "outcome_names", ["sales"])
    if isinstance(outcome_names, str):
        outcome_names = [outcome_names]
    
    # Helper to check if variable exists in posterior
    def _var_exists(var_name: str) -> bool:
        try:
            if hasattr(posterior, 'data_vars'):
                return var_name in posterior.data_vars
            return var_name in posterior
        except Exception:
            return False
    
    # Helper to get samples from posterior
    def _get_samples(var_name: str) -> np.ndarray | None:
        try:
            if _var_exists(var_name):
                return _flatten_samples(posterior[var_name].values)
        except Exception as e:
            logger.debug(f"Could not get samples for {var_name}: {e}")
        return None
    
    for channel in channels:
        for outcome in outcome_names:
            # Look for direct/indirect/total deterministics
            direct_var = f"direct_{channel}_{outcome}"
            indirect_var = f"indirect_{channel}_{outcome}"
            total_var = f"total_{channel}_{outcome}"
            
            direct_samples = _get_samples(direct_var)
            total_samples = _get_samples(total_var)
            
            if direct_samples is not None and total_samples is not None:
                indirect_samples = _get_samples(indirect_var)
                if indirect_samples is None:
                    indirect_samples = total_samples - direct_samples
                
                direct_mean = float(np.mean(direct_samples))
                direct_lower, direct_upper = _compute_hdi(direct_samples, hdi_prob)
                
                indirect_mean = float(np.mean(indirect_samples))
                indirect_lower, indirect_upper = _compute_hdi(indirect_samples, hdi_prob)
                
                total_mean = float(np.mean(total_samples))
                total_lower, total_upper = _compute_hdi(total_samples, hdi_prob)
                
                prop_mediated = indirect_mean / total_mean if total_mean != 0 else 0.0
                
                results.append(MediatedEffectResult(
                    channel=channel,
                    outcome=outcome,
                    direct_mean=direct_mean,
                    direct_lower=direct_lower,
                    direct_upper=direct_upper,
                    indirect_mean=indirect_mean,
                    indirect_lower=indirect_lower,
                    indirect_upper=indirect_upper,
                    total_mean=total_mean,
                    total_lower=total_lower,
                    total_upper=total_upper,
                    proportion_mediated=prop_mediated,
                ))
    
    return results


def compute_cross_effects(
    model: Any,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """
    Compute cross-effects between outcomes for multivariate models.
    
    For models with multiple outcomes (e.g., Product A sales, Product B sales),
    this extracts the cross-effect coefficients showing how one outcome
    influences another.
    
    Parameters
    ----------
    model : MultivariateMMM or CombinedMMM
        Fitted multivariate model
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    pd.DataFrame
        Cross-effect matrix with uncertainty
    """
    _check_model_fitted(model)
    
    if not hasattr(model, "outcome_names"):
        raise ValueError("Model does not support cross-effect analysis")
    
    # Try model's built-in method
    if hasattr(model, "get_cross_effect_summary"):
        try:
            result = model.get_cross_effect_summary()
            if result is not None and len(result) > 0:
                return result
        except Exception as e:
            logger.warning(f"Model cross-effect method failed: {e}")
    
    # Manual computation from trace
    posterior = _get_posterior(model)
    outcome_names = model.outcome_names
    
    if posterior is None:
        logger.warning("No posterior found for cross-effect computation")
        return pd.DataFrame()
    
    rows = []
    
    # Helper to check if variable exists in posterior
    def _var_exists(var_name: str) -> bool:
        try:
            if hasattr(posterior, 'data_vars'):
                return var_name in posterior.data_vars
            return var_name in posterior
        except Exception:
            return False
    
    # Look for psi matrix or psi_matrix (different naming conventions)
    psi_var_name = None
    for name in ["psi", "psi_matrix", "cross_effects"]:
        if _var_exists(name):
            psi_var_name = name
            break
    
    if psi_var_name is not None:
        try:
            psi_samples = posterior[psi_var_name].values
            
            # Handle different shapes - could be (chain, draw, outcome, outcome) 
            # or just (samples, outcome, outcome)
            if psi_samples.ndim == 4:
                # (chain, draw, outcome, outcome)
                for i, source in enumerate(outcome_names):
                    for j, target in enumerate(outcome_names):
                        if i != j:
                            samples = psi_samples[:, :, i, j].flatten()
                            mean = float(np.mean(samples))
                            lower, upper = _compute_hdi(samples, hdi_prob)
                            
                            rows.append({
                                "source": source,
                                "target": target,
                                "effect_mean": mean,
                                "effect_hdi_low": lower,
                                "effect_hdi_high": upper,
                                "prob_positive": float(np.mean(samples > 0)),
                            })
            elif psi_samples.ndim == 3:
                # (samples, outcome, outcome)
                for i, source in enumerate(outcome_names):
                    for j, target in enumerate(outcome_names):
                        if i != j:
                            samples = psi_samples[:, i, j].flatten()
                            mean = float(np.mean(samples))
                            lower, upper = _compute_hdi(samples, hdi_prob)
                            
                            rows.append({
                                "source": source,
                                "target": target,
                                "effect_mean": mean,
                                "effect_hdi_low": lower,
                                "effect_hdi_high": upper,
                                "prob_positive": float(np.mean(samples > 0)),
                            })
        except Exception as e:
            logger.warning(f"Error extracting cross-effects from {psi_var_name}: {e}")
    
    return pd.DataFrame(rows)


# =============================================================================
# Summary Report Generation
# =============================================================================


def generate_model_summary(
    model: Any,
    hdi_prob: float = 0.94,
) -> dict[str, Any]:
    """
    Generate comprehensive model summary for reporting.
    
    Aggregates key metrics into a single dictionary suitable for
    report generation or dashboard display.
    
    Parameters
    ----------
    model : BayesianMMM or ExtendedMMM
        Fitted model
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    dict
        Summary containing:
        - model_info: Basic model metadata
        - diagnostics: MCMC convergence diagnostics
        - roi_summary: ROI by channel
        - decomposition: Component contributions
        - saturation_summary: Saturation levels
        - adstock_summary: Carryover effects
    """
    _check_model_fitted(model)
    
    summary = {
        "model_info": _get_model_info(model),
        "diagnostics": _get_diagnostics(model),
    }
    
    # ROI
    try:
        roi_df = compute_roi_with_uncertainty(model, hdi_prob=hdi_prob)
        summary["roi_summary"] = roi_df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"ROI computation failed: {e}")
        summary["roi_summary"] = None
    
    # Decomposition
    try:
        decomp = compute_component_decomposition(model, include_time_series=False, hdi_prob=hdi_prob)
        summary["decomposition"] = [d.to_dict() for d in decomp]
    except Exception as e:
        logger.warning(f"Decomposition failed: {e}")
        summary["decomposition"] = None
    
    # Saturation
    try:
        sat_curves = compute_saturation_curves_with_uncertainty(model, n_points=50, hdi_prob=hdi_prob)
        summary["saturation_summary"] = {
            ch: {
                "saturation_level": curve.saturation_level,
                "marginal_response": curve.marginal_response_at_current,
            }
            for ch, curve in sat_curves.items()
        }
    except Exception as e:
        logger.warning(f"Saturation computation failed: {e}")
        summary["saturation_summary"] = None
    
    # Adstock
    try:
        adstock = compute_adstock_weights(model, hdi_prob=hdi_prob)
        summary["adstock_summary"] = {
            ch: {
                "half_life": result.half_life,
                "total_carryover": result.total_carryover,
                "alpha_mean": result.alpha_mean,
            }
            for ch, result in adstock.items()
        }
    except Exception as e:
        logger.warning(f"Adstock computation failed: {e}")
        summary["adstock_summary"] = None
    
    return summary


def _get_model_info(model: Any) -> dict[str, Any]:
    """Extract basic model info."""
    info = {
        "model_type": type(model).__name__,
        "n_obs": getattr(model, "n_obs", None),
        "n_channels": getattr(model, "n_channels", len(_get_channel_names(model))),
        "channel_names": _get_channel_names(model),
    }
    
    # Add geo/product info if available
    if hasattr(model, "has_geo"):
        info["has_geo"] = model.has_geo
        info["n_geos"] = getattr(model, "n_geos", None)
    
    if hasattr(model, "has_product"):
        info["has_product"] = model.has_product
        info["n_products"] = getattr(model, "n_products", None)
    
    # Extended model info
    if hasattr(model, "mediator_names"):
        info["mediator_names"] = list(model.mediator_names)
    if hasattr(model, "outcome_names"):
        info["outcome_names"] = list(model.outcome_names)
    
    return info


def _get_diagnostics(model: Any) -> dict[str, Any]:
    """Extract MCMC diagnostics."""
    trace = _get_trace(model)
    
    if trace is None:
        return {}
    
    diagnostics = {}
    
    try:
        if az is not None:
            summary = az.summary(trace)
            diagnostics["rhat_max"] = float(summary["r_hat"].max())
            diagnostics["ess_bulk_min"] = float(summary["ess_bulk"].min())
            diagnostics["ess_tail_min"] = float(summary["ess_tail"].min())
        
        # Check for divergences
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            diagnostics["divergences"] = int(trace.sample_stats["diverging"].values.sum())
        else:
            diagnostics["divergences"] = 0
        
        # Convergence status
        diagnostics["converged"] = (
            diagnostics.get("divergences", 0) == 0 and
            diagnostics.get("rhat_max", 2.0) < 1.01 and
            diagnostics.get("ess_bulk_min", 0) > 400
        )
        
    except Exception as e:
        logger.warning(f"Error extracting diagnostics: {e}")
    
    return diagnostics


# =============================================================================
# Export Functions for Integration
# =============================================================================


__all__ = [
    # Result containers
    "ROIResult",
    "PriorPosteriorComparison",
    "SaturationCurveResult",
    "AdstockResult",
    "DecompositionResult",
    "MediatedEffectResult",
    # ROI functions
    "compute_roi_with_uncertainty",
    "compute_marginal_roi",
    # Prior/posterior comparison
    "get_prior_posterior_comparison",
    "compute_shrinkage_summary",
    # Saturation
    "compute_saturation_curves_with_uncertainty",
    # Adstock
    "compute_adstock_weights",
    # Decomposition
    "compute_component_decomposition",
    "compute_decomposition_waterfall",
    # Extended models
    "compute_mediated_effects",
    "compute_cross_effects",
    # Summary
    "generate_model_summary",
]