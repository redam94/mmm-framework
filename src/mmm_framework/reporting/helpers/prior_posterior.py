"""
Prior vs posterior comparison functions for MMM reporting.

Functions for comparing prior and posterior distributions to assess
data informativeness and parameter shrinkage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import pymc as pm
except ImportError:
    pm = None

from .results import PriorPosteriorComparison
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
)


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

        results.append(
            PriorPosteriorComparison(
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
            )
        )

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
        rows.append(
            {
                "parameter": c.parameter,
                "prior_mean": c.prior_mean,
                "prior_sd": c.prior_sd,
                "posterior_mean": c.posterior_mean,
                "posterior_sd": c.posterior_sd,
                "shrinkage": c.shrinkage,
                "data_informative": (
                    "Yes"
                    if c.shrinkage and c.shrinkage > 0.5
                    else "No" if c.shrinkage else "Unknown"
                ),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "get_prior_posterior_comparison",
    "compute_shrinkage_summary",
    "_select_key_parameters",
]
