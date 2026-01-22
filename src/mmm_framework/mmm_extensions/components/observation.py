"""
Observation model builders for MMM Extensions.

These functions build various observation/likelihood models
for different types of data (Gaussian, partial, multivariate, survey).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    from ..config import AggregatedSurveyConfig, MediatorConfigExtended


def build_gaussian_likelihood(
    name: str,
    mu: pt.TensorVariable,
    observed: np.ndarray,
    sigma_prior_sigma: float = 0.5,
    dims: str | None = None,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """
    Build Gaussian likelihood.

    Parameters
    ----------
    name : str
        Variable name
    mu : TensorVariable
        Expected value
    observed : np.ndarray
        Observed data
    sigma_prior_sigma : float
        Prior on observation noise
    dims : str | None
        PyMC dimension

    Returns
    -------
    tuple
        (likelihood_rv, sigma_rv)
    """
    sigma = pm.HalfNormal(f"{name}_sigma", sigma=sigma_prior_sigma)

    kwargs = {"observed": observed}
    if dims:
        kwargs["dims"] = dims

    y_obs = pm.Normal(name, mu=mu, sigma=sigma, **kwargs)
    return y_obs, sigma


def build_partial_observation_model(
    name: str,
    latent: pt.TensorVariable,
    observed: np.ndarray,
    mask: np.ndarray,
    sigma_prior_sigma: float = 0.1,
) -> tuple[pt.TensorVariable | None, pt.TensorVariable]:
    """
    Build observation model for partially observed variable.

    Parameters
    ----------
    name : str
        Variable name
    latent : TensorVariable
        Latent true values
    observed : np.ndarray
        Observed values (with NaN for missing)
    mask : np.ndarray
        Boolean mask (True = observed)
    sigma_prior_sigma : float
        Prior on measurement noise

    Returns
    -------
    tuple
        (likelihood_rv or None, sigma_rv)
    """
    sigma = pm.HalfNormal(f"{name}_obs_sigma", sigma=sigma_prior_sigma)

    if mask.any():
        obs_rv = pm.Normal(
            f"{name}_observed",
            mu=latent[mask],
            sigma=sigma,
            observed=observed[mask],
        )
        return obs_rv, sigma

    return None, sigma


def build_multivariate_likelihood(
    name: str,
    mu: pt.TensorVariable,
    observed: np.ndarray,
    n_outcomes: int,
    lkj_eta: float = 2.0,
    sigma_prior_sigma: float = 0.5,
    dims: tuple[str, str] | None = None,
) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """
    Build multivariate normal likelihood with LKJ correlation prior.

    Parameters
    ----------
    name : str
        Variable name
    mu : TensorVariable
        Expected values (n_obs, n_outcomes)
    observed : np.ndarray
        Observed data
    n_outcomes : int
        Number of outcomes
    lkj_eta : float
        LKJ prior concentration parameter
    sigma_prior_sigma : float
        Prior on outcome standard deviations
    dims : tuple | None
        PyMC dimensions

    Returns
    -------
    tuple
        (likelihood_rv, cholesky_factor, correlation_matrix)
    """
    chol, corr, _ = pm.LKJCholeskyCov(
        f"{name}_chol_cov",
        n=n_outcomes,
        eta=lkj_eta,
        sd_dist=pm.HalfNormal.dist(sigma=sigma_prior_sigma),
        compute_corr=True,
    )

    pm.Deterministic(f"{name}_correlation", corr)

    kwargs = {"observed": observed}
    if dims:
        kwargs["dims"] = dims

    y_obs = pm.MvNormal(name, mu=mu, chol=chol, **kwargs)
    return y_obs, chol, corr


def build_aggregated_survey_observation(
    name: str,
    latent: pt.TensorVariable,
    observed_data: np.ndarray,
    config: "AggregatedSurveyConfig",
    is_proportion: bool = True,
) -> None:
    """
    Build observation model for temporally aggregated survey data.

    This handles the case where:
    1. Surveys are fielded continuously over a period (e.g., month)
    2. Results are aggregated across that period
    3. Sample size varies by wave (heteroskedastic observation noise)

    Parameters
    ----------
    name : str
        Name prefix for PyMC variables.
    latent : TensorVariable
        Latent mediator values at model frequency (e.g., weekly).
    observed_data : np.ndarray
        Observed survey results.
    config : AggregatedSurveyConfig
        Configuration for the aggregated observation model.
    is_proportion : bool
        If True and using binomial, observed_data is proportions.
    """
    from ..config import AggregatedSurveyLikelihood

    # Compute aggregated latent values
    aggregated_latent = []
    for obs_idx in sorted(config.aggregation_map.keys()):
        constituent_indices = list(config.aggregation_map[obs_idx])

        if config.aggregation_function == "mean":
            agg_value = latent[constituent_indices].mean()
        elif config.aggregation_function == "sum":
            agg_value = latent[constituent_indices].sum()
        elif config.aggregation_function == "last":
            agg_value = latent[constituent_indices[-1]]
        else:
            raise ValueError(f"Unknown aggregation function: {config.aggregation_function}")

        aggregated_latent.append(agg_value)

    # Stack into tensor
    mu = pt.stack(aggregated_latent)

    # Sample sizes as array
    n = np.array(config.sample_sizes)

    # Effective sample size (accounting for design effect)
    n_eff = n / config.design_effect

    # Build likelihood based on config
    if config.likelihood == AggregatedSurveyLikelihood.BINOMIAL:
        _build_binomial_observation(name, mu, observed_data, n, is_proportion)

    elif config.likelihood == AggregatedSurveyLikelihood.NORMAL:
        _build_normal_observation(name, mu, observed_data, n_eff, is_proportion)

    elif config.likelihood == AggregatedSurveyLikelihood.BETA_BINOMIAL:
        _build_beta_binomial_observation(
            name, mu, observed_data, n,
            is_proportion, config.overdispersion_prior_sigma
        )

    else:
        raise ValueError(f"Unknown likelihood: {config.likelihood}")


def _build_binomial_observation(
    name: str,
    mu: pt.TensorVariable,
    observed_data: np.ndarray,
    sample_sizes: np.ndarray,
    is_proportion: bool,
) -> None:
    """Exact binomial likelihood for survey observations."""
    if is_proportion:
        observed_counts = np.round(observed_data * sample_sizes).astype(int)
    else:
        observed_counts = observed_data.astype(int)

    # Clamp latent probability to valid range
    p = pt.clip(mu, 1e-6, 1 - 1e-6)

    pm.Binomial(
        f"{name}_obs",
        n=sample_sizes,
        p=p,
        observed=observed_counts,
    )


def _build_normal_observation(
    name: str,
    mu: pt.TensorVariable,
    observed_data: np.ndarray,
    n_eff: np.ndarray,
    is_proportion: bool,
) -> None:
    """Normal approximation with sample-size-dependent standard errors."""
    if not is_proportion:
        raise ValueError("Normal likelihood requires proportion data")

    # Standard error from binomial sampling: SE = sqrt(p*(1-p)/n)
    p_obs = np.clip(observed_data, 0.01, 0.99)
    sampling_se = np.sqrt(p_obs * (1 - p_obs) / n_eff)

    pm.Normal(
        f"{name}_obs",
        mu=mu,
        sigma=sampling_se,
        observed=observed_data,
    )


def _build_beta_binomial_observation(
    name: str,
    mu: pt.TensorVariable,
    observed_data: np.ndarray,
    sample_sizes: np.ndarray,
    is_proportion: bool,
    overdispersion_prior_sigma: float,
) -> None:
    """Beta-binomial likelihood for overdispersed survey data."""
    if is_proportion:
        observed_counts = np.round(observed_data * sample_sizes).astype(int)
    else:
        observed_counts = observed_data.astype(int)

    # Clamp mean probability
    p = pt.clip(mu, 1e-6, 1 - 1e-6)

    # Overdispersion parameter (concentration)
    kappa = pm.HalfNormal(f"{name}_kappa", sigma=1/overdispersion_prior_sigma)

    # Reparameterize: alpha = p * kappa, beta = (1-p) * kappa
    alpha = p * kappa
    beta = (1 - p) * kappa

    pm.BetaBinomial(
        f"{name}_obs",
        n=sample_sizes,
        alpha=alpha,
        beta=beta,
        observed=observed_counts,
    )


def compute_survey_observation_indices(
    model_frequency: str,
    survey_frequency: str,
    n_periods: int,
    start_date: str | None = None,
) -> dict[int, tuple[int, ...]]:
    """
    Compute aggregation map from model and survey frequencies.

    Parameters
    ----------
    model_frequency : str
        Model time frequency: "daily", "weekly"
    survey_frequency : str
        Survey aggregation frequency: "weekly", "monthly", "quarterly"
    n_periods : int
        Number of model periods.
    start_date : str, optional
        Start date for calendar-based aggregation (ISO format).

    Returns
    -------
    dict[int, tuple[int, ...]]
        Aggregation map suitable for AggregatedSurveyConfig.
    """
    if start_date is not None:
        # Calendar-based aggregation
        import pandas as pd
        dates = pd.date_range(
            start=start_date, periods=n_periods, freq=model_frequency[0].upper()
        )

        if survey_frequency == "monthly":
            groups = dates.to_period("M")
        elif survey_frequency == "quarterly":
            groups = dates.to_period("Q")
        elif survey_frequency == "weekly":
            groups = dates.to_period("W")
        else:
            raise ValueError(f"Unknown survey frequency: {survey_frequency}")

        aggregation_map = {}
        for survey_idx, period in enumerate(groups.unique()):
            mask = groups == period
            indices = tuple(np.where(mask)[0])
            aggregation_map[survey_idx] = indices

        return aggregation_map

    else:
        # Simple numeric grouping
        if model_frequency == "weekly" and survey_frequency == "monthly":
            periods_per_survey = 4
        elif model_frequency == "weekly" and survey_frequency == "quarterly":
            periods_per_survey = 13
        elif model_frequency == "daily" and survey_frequency == "weekly":
            periods_per_survey = 7
        elif model_frequency == "daily" and survey_frequency == "monthly":
            periods_per_survey = 30
        else:
            raise ValueError(
                f"Unsupported frequency combination: {model_frequency} -> {survey_frequency}"
            )

        n_surveys = n_periods // periods_per_survey
        aggregation_map = {}

        for survey_idx in range(n_surveys):
            start = survey_idx * periods_per_survey
            end = start + periods_per_survey
            aggregation_map[survey_idx] = tuple(range(start, min(end, n_periods)))

        return aggregation_map


def build_mediator_observation_dispatch(
    med_config: "MediatorConfigExtended",
    mediator_latent: pt.TensorVariable,
    mediator_data: dict[str, np.ndarray],
    mediator_masks: dict[str, np.ndarray],
) -> None:
    """
    Dispatch to appropriate observation model based on mediator type.

    Parameters
    ----------
    med_config : MediatorConfigExtended
        Mediator configuration.
    mediator_latent : TensorVariable
        Latent mediator values.
    mediator_data : dict[str, np.ndarray]
        Observed mediator data.
    mediator_masks : dict[str, np.ndarray]
        Observation masks for partial observation.
    """
    from ..config import MediatorObservationType

    med_name = med_config.name
    obs_type = med_config.observation_type

    if obs_type == MediatorObservationType.FULLY_LATENT:
        # No observation model
        return

    if med_name not in mediator_data:
        # No data provided
        return

    obs_data = mediator_data[med_name]

    if obs_type == MediatorObservationType.FULLY_OBSERVED:
        # Every period observed with constant noise
        pm.Normal(
            f"{med_name}_obs",
            mu=mediator_latent,
            sigma=med_config.observation_noise_sigma,
            observed=obs_data,
        )

    elif obs_type == MediatorObservationType.PARTIALLY_OBSERVED:
        # Sparse point-in-time observations
        mask = mediator_masks.get(med_name, ~np.isnan(obs_data))
        pm.Normal(
            f"{med_name}_obs",
            mu=mediator_latent[mask],
            sigma=med_config.observation_noise_sigma,
            observed=obs_data[mask],
        )

    elif obs_type == MediatorObservationType.AGGREGATED_SURVEY:
        # Aggregated survey with sample-size-dependent noise
        if med_config.aggregated_survey_config is None:
            raise ValueError(
                f"aggregated_survey_config required for {med_name}"
            )
        build_aggregated_survey_observation(
            name=med_name,
            latent=mediator_latent,
            observed_data=obs_data,
            config=med_config.aggregated_survey_config,
            is_proportion=True,
        )

    else:
        raise ValueError(f"Unknown observation type: {obs_type}")


__all__ = [
    "build_gaussian_likelihood",
    "build_partial_observation_model",
    "build_multivariate_likelihood",
    "build_aggregated_survey_observation",
    "compute_survey_observation_indices",
    "build_mediator_observation_dispatch",
]
