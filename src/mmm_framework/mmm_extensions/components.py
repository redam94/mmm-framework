"""
Base Components for MMM Extensions

Atomic, composable building blocks for nested and multivariate models.
These components are designed to be reused across different model types.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pymc as pm
from pytensor import scan as pytensor_scan
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Callable, Any
from enum import Enum

if TYPE_CHECKING:
    import pandas as pd
    import arviz as az

from .config import (
    VariableSelectionMethod,
    HorseshoeConfig,
    SpikeSlabConfig,
    LassoConfig,
    VariableSelectionConfig,
)

# =============================================================================
# Type Definitions
# =============================================================================


class TransformFn(Protocol):
    """Protocol for transformation functions."""

    def __call__(self, x: pt.TensorVariable, **params) -> pt.TensorVariable: ...


# =============================================================================
# Atomic Transformation Functions
# =============================================================================


def geometric_adstock_np(
    x: np.ndarray,
    alpha: float,
    l_max: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply geometric adstock transformation (NumPy version).

    Use this for data preprocessing before model building.

    Parameters
    ----------
    x : np.ndarray
        Input media variable (n_obs,)
    alpha : float
        Decay rate [0, 1]
    l_max : int
        Maximum lag length
    normalize : bool
        Whether to normalize weights to sum to 1

    Returns
    -------
    np.ndarray
        Adstocked media variable
    """
    weights = np.power(alpha, np.arange(l_max))
    if normalize:
        weights = weights / weights.sum()

    # Convolve with zero-padding
    result = np.convolve(x, weights[::-1], mode="full")[: len(x)]
    return result


def geometric_adstock_pt(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    l_max: int = 8,
    normalize: bool = True,
) -> pt.TensorVariable:
    """
    Apply geometric adstock transformation (PyTensor version with scan).

    Parameters
    ----------
    x : TensorVariable
        Input media variable (n_obs,)
    alpha : TensorVariable
        Decay rate [0, 1]
    l_max : int
        Maximum lag length
    normalize : bool
        Whether to normalize weights to sum to 1

    Returns
    -------
    TensorVariable
        Adstocked media variable
    """
    # Build weights
    weights = pt.power(alpha, pt.arange(l_max))
    if normalize:
        weights = weights / weights.sum()

    # Use scan for proper gradient flow
    def step(x_t, carry, w):
        # Shift carry and add new value
        new_carry = pt.concatenate([[x_t], carry[:-1]])
        # Weighted sum
        y_t = pt.dot(new_carry, w)
        return y_t, new_carry

    # Initial carry (zeros)
    init_carry = pt.zeros(l_max)

    outputs, _ = pytensor_scan(
        fn=step,
        sequences=[x],
        outputs_info=[None, init_carry],
        non_sequences=[weights],
    )

    return outputs[0]


def geometric_adstock_convolution(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    l_max: int = 8,
    normalize: bool = True,
) -> pt.TensorVariable:
    """
    Apply geometric adstock using matrix multiplication (no scan).

    This is often more efficient and avoids scan complexity.
    Requires knowing n_obs at graph construction time.

    Parameters
    ----------
    x : TensorVariable
        Input media variable (n_obs,)
    alpha : TensorVariable
        Decay rate [0, 1]
    l_max : int
        Maximum lag length
    normalize : bool
        Whether to normalize weights to sum to 1

    Returns
    -------
    TensorVariable
        Adstocked media variable
    """
    # Build weights
    weights = pt.power(alpha, pt.arange(l_max))
    if normalize:
        weights = weights / weights.sum()

    # Pad input
    x_padded = pt.concatenate([pt.zeros(l_max - 1), x])

    # Build convolution using indexing (simpler than scan)
    n = x.shape[0]

    # Create indices for convolution
    # Result[i] = sum(x_padded[i:i+l_max] * weights[::-1])
    indices = pt.arange(l_max)

    def convolve_at(i):
        window = x_padded[i : i + l_max]
        return pt.dot(window, weights[::-1])

    # Vectorized using advanced indexing
    # Build a matrix where each row is a window
    row_indices = pt.arange(n)[:, None] + indices[None, :]
    windows = x_padded[row_indices]  # (n, l_max)

    return pt.dot(windows, weights[::-1])


# Alias for most common use case
geometric_adstock = geometric_adstock_convolution


def geometric_adstock_matrix(
    X: pt.TensorVariable,
    alphas: pt.TensorVariable,
    l_max: int = 8,
) -> pt.TensorVariable:
    """
    Apply geometric adstock to multiple channels.

    Parameters
    ----------
    X : TensorVariable
        Media matrix (n_obs, n_channels)
    alphas : TensorVariable
        Decay rates per channel (n_channels,)
    l_max : int
        Maximum lag length

    Returns
    -------
    TensorVariable
        Adstocked media matrix (n_obs, n_channels)
    """
    n_channels = X.shape[1].eval().astype(int)
    results = []
    for i in range(n_channels):
        results.append(geometric_adstock(X[:, i], alphas[i], l_max))
    return pt.stack(results, axis=1)


def logistic_saturation(
    x: pt.TensorVariable,
    lam: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Apply logistic saturation transformation.

    Parameters
    ----------
    x : TensorVariable
        Input (already adstocked)
    lam : TensorVariable
        Saturation rate (higher = faster saturation)

    Returns
    -------
    TensorVariable
        Saturated output in [0, 1]
    """
    return 1 - pt.exp(-lam * x)


def hill_saturation(
    x: pt.TensorVariable,
    kappa: pt.TensorVariable,
    slope: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Apply Hill saturation transformation.

    Parameters
    ----------
    x : TensorVariable
        Input (already adstocked)
    kappa : TensorVariable
        Half-saturation point (EC50)
    slope : TensorVariable
        Steepness of curve

    Returns
    -------
    TensorVariable
        Saturated output in [0, 1]
    """
    x_safe = pt.maximum(x, 1e-10)
    return pt.power(x_safe, slope) / (pt.power(kappa, slope) + pt.power(x_safe, slope))


def apply_transformation_pipeline(
    x: pt.TensorVariable,
    transforms: list[tuple[Callable, dict[str, Any]]],
) -> pt.TensorVariable:
    """
    Apply a sequence of transformations.

    Parameters
    ----------
    x : TensorVariable
        Input variable
    transforms : list[tuple[Callable, dict]]
        List of (transform_fn, params_dict) tuples

    Returns
    -------
    TensorVariable
        Transformed output
    """
    result = x
    for fn, params in transforms:
        result = fn(result, **params)
    return result


# =============================================================================
# Prior Factory Functions
# =============================================================================


def create_adstock_prior(
    name: str,
    prior_type: str = "beta",
    **kwargs,
) -> pt.TensorVariable:
    """
    Create adstock decay prior.

    Parameters
    ----------
    name : str
        Parameter name
    prior_type : str
        "beta" or "uniform"
    **kwargs
        Additional prior parameters

    Returns
    -------
    TensorVariable
        Prior random variable
    """
    if prior_type == "beta":
        alpha = kwargs.get("alpha", 2)
        beta = kwargs.get("beta", 2)
        return pm.Beta(name, alpha=alpha, beta=beta)
    elif prior_type == "uniform":
        return pm.Uniform(name, lower=0, upper=1)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


def create_saturation_prior(
    name: str,
    saturation_type: str = "logistic",
    **kwargs,
) -> dict[str, pt.TensorVariable]:
    """
    Create saturation parameter priors.

    Parameters
    ----------
    name : str
        Base parameter name
    saturation_type : str
        "logistic" or "hill"
    **kwargs
        Prior hyperparameters

    Returns
    -------
    dict
        Dictionary of prior random variables
    """
    if saturation_type == "logistic":
        lam_alpha = kwargs.get("lam_alpha", 3)
        lam_beta = kwargs.get("lam_beta", 1)
        return {"lam": pm.Gamma(f"{name}_lam", alpha=lam_alpha, beta=lam_beta)}
    elif saturation_type == "hill":
        return {
            "kappa": pm.Beta(f"{name}_kappa", alpha=2, beta=2),
            "slope": pm.Gamma(f"{name}_slope", alpha=3, beta=1),
        }
    else:
        raise ValueError(f"Unknown saturation type: {saturation_type}")


def create_effect_prior(
    name: str,
    constrained: str = "none",
    mu: float = 0.0,
    sigma: float = 1.0,
    dims: str | tuple | None = None,
) -> pt.TensorVariable:
    """
    Create effect coefficient prior.

    Parameters
    ----------
    name : str
        Parameter name
    constrained : str
        "none", "positive", "negative"
    mu : float
        Prior mean (for unconstrained)
    sigma : float
        Prior scale
    dims : str | tuple | None
        PyMC dimensions

    Returns
    -------
    TensorVariable
        Prior random variable
    """
    kwargs = {"sigma": sigma}
    if dims is not None:
        kwargs["dims"] = dims

    if constrained == "positive":
        return pm.HalfNormal(name, **kwargs)
    elif constrained == "negative":
        return -pm.HalfNormal(name, **kwargs)
    else:
        kwargs["mu"] = mu
        return pm.Normal(name, **kwargs)


# =============================================================================
# Model Component Builders
# =============================================================================


@dataclass
class MediaTransformResult:
    """Result of media transformation."""

    transformed: pt.TensorVariable  # (n_obs, n_channels)
    adstock_params: dict[str, pt.TensorVariable]
    saturation_params: dict[str, pt.TensorVariable]


def build_media_transforms(
    X_media: pt.TensorVariable,
    channel_names: list[str],
    adstock_config: dict[str, Any],
    saturation_config: dict[str, Any],
    share_params: bool = False,
    name_prefix: str = "",
) -> MediaTransformResult:
    """
    Build media transformation block.

    Parameters
    ----------
    X_media : TensorVariable
        Raw media matrix (n_obs, n_channels)
    channel_names : list[str]
        Channel names
    adstock_config : dict
        Adstock configuration
    saturation_config : dict
        Saturation configuration
    share_params : bool
        Whether to share parameters across channels
    name_prefix : str
        Prefix for parameter names

    Returns
    -------
    MediaTransformResult
        Transformed media and parameters
    """
    n_channels = len(channel_names)
    prefix = f"{name_prefix}_" if name_prefix else ""

    adstock_params = {}
    saturation_params = {}
    transformed_channels = []

    # Create adstock parameters
    if share_params:
        alpha = create_adstock_prior(
            f"{prefix}alpha_shared",
            prior_type=adstock_config.get("prior_type", "beta"),
            **adstock_config.get("prior_params", {}),
        )
        adstock_params["shared"] = alpha

    # Transform each channel
    for i, channel in enumerate(channel_names):
        x = X_media[:, i]

        # Adstock
        if share_params:
            alpha = adstock_params["shared"]
        else:
            alpha = create_adstock_prior(
                f"{prefix}alpha_{channel}",
                prior_type=adstock_config.get("prior_type", "beta"),
                **adstock_config.get("prior_params", {}),
            )
            adstock_params[channel] = alpha

        l_max = adstock_config.get("l_max", 8)
        x_adstocked = geometric_adstock(x, alpha, l_max)

        # Saturation
        sat_type = saturation_config.get("type", "logistic")
        sat_params = create_saturation_prior(
            f"{prefix}sat_{channel}",
            saturation_type=sat_type,
            **saturation_config.get("prior_params", {}),
        )
        saturation_params[channel] = sat_params

        if sat_type == "logistic":
            x_saturated = logistic_saturation(x_adstocked, sat_params["lam"])
        else:
            x_saturated = hill_saturation(
                x_adstocked, sat_params["kappa"], sat_params["slope"]
            )

        transformed_channels.append(x_saturated)

    transformed = pt.stack(transformed_channels, axis=1)

    return MediaTransformResult(
        transformed=transformed,
        adstock_params=adstock_params,
        saturation_params=saturation_params,
    )


@dataclass
class EffectResult:
    """Result of effect computation."""

    contribution: pt.TensorVariable  # (n_obs,)
    coefficients: pt.TensorVariable
    components: pt.TensorVariable | None = None  # (n_obs, n_vars) if multiple


def build_linear_effect(
    X: pt.TensorVariable,
    var_names: list[str],
    name_prefix: str,
    constrained: str = "none",
    prior_sigma: float = 0.5,
    dims: str | None = None,
) -> EffectResult:
    """
    Build linear effect block.

    Parameters
    ----------
    X : TensorVariable
        Design matrix (n_obs, n_vars)
    var_names : list[str]
        Variable names
    name_prefix : str
        Prefix for parameter names
    constrained : str
        Constraint type
    prior_sigma : float
        Prior scale
    dims : str | None
        Dimension name for coefficients

    Returns
    -------
    EffectResult
        Effect contribution and coefficients
    """
    n_vars = len(var_names)

    # Create coefficient prior
    beta = create_effect_prior(
        f"{name_prefix}_beta",
        constrained=constrained,
        sigma=prior_sigma,
        dims=dims,
    )

    # Compute contribution
    if n_vars == 1:
        contribution = beta * X[:, 0]
        components = None
    else:
        components = X * beta  # Broadcasting
        contribution = components.sum(axis=1)

    return EffectResult(
        contribution=contribution,
        coefficients=beta,
        components=components,
    )


# =============================================================================
# Result Container
# =============================================================================


@dataclass
class VariableSelectionResult:
    """
    Container for variable selection prior outputs.

    Attributes
    ----------
    beta : pt.TensorVariable
        The coefficient vector with shrinkage/selection applied.
    inclusion_indicators : pt.TensorVariable | None
        For spike-slab: soft inclusion indicators (gamma).
    local_shrinkage : pt.TensorVariable | None
        For horseshoe: local shrinkage parameters (lambda).
    global_shrinkage : pt.TensorVariable | None
        For horseshoe: global shrinkage parameter (tau).
    effective_nonzero : pt.TensorVariable | None
        Estimated number of effectively nonzero coefficients.
    kappa : pt.TensorVariable | None
        Shrinkage factors for each coefficient (horseshoe).
    """

    beta: pt.TensorVariable
    inclusion_indicators: pt.TensorVariable | None = None
    local_shrinkage: pt.TensorVariable | None = None
    global_shrinkage: pt.TensorVariable | None = None
    effective_nonzero: pt.TensorVariable | None = None
    kappa: pt.TensorVariable | None = None


# =============================================================================
# Observation Model Builders
# =============================================================================


def build_gaussian_likelihood(
    name: str,
    mu: pt.TensorVariable,
    observed: np.ndarray,
    sigma_prior_sigma: float = 0.5,
    dims: str | None = None,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """
    Build Gaussian likelihood.

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


# =============================================================================
# Cross-Effect Builders
# =============================================================================


@dataclass
class CrossEffectSpec:
    """Specification for a single cross-effect."""

    source_idx: int
    target_idx: int
    effect_type: str  # "cannibalization", "halo", "unconstrained"
    prior_sigma: float = 0.3


def build_cross_effect_matrix(
    specs: list[CrossEffectSpec],
    n_outcomes: int,
    name_prefix: str = "psi",
) -> tuple[pt.TensorVariable, dict[tuple[int, int], pt.TensorVariable]]:
    """
    Build cross-effect coefficient matrix.

    Parameters
    ----------
    specs : list[CrossEffectSpec]
        Cross-effect specifications
    n_outcomes : int
        Number of outcomes
    name_prefix : str
        Prefix for parameter names

    Returns
    -------
    tuple
        (cross_effect_matrix, individual_params_dict)
    """
    psi_matrix = pt.zeros((n_outcomes, n_outcomes))
    params = {}

    for spec in specs:
        param_name = f"{name_prefix}_{spec.source_idx}_{spec.target_idx}"

        if spec.effect_type == "cannibalization":
            # Negative effect
            psi_raw = pm.HalfNormal(f"{param_name}_raw", sigma=spec.prior_sigma)
            psi = -psi_raw
        elif spec.effect_type == "halo":
            # Positive effect
            psi = pm.HalfNormal(param_name, sigma=spec.prior_sigma)
        else:
            # Unconstrained
            psi = pm.Normal(param_name, mu=0, sigma=spec.prior_sigma)

        psi_matrix = pt.set_subtensor(psi_matrix[spec.source_idx, spec.target_idx], psi)
        params[(spec.source_idx, spec.target_idx)] = psi

    return psi_matrix, params


def compute_cross_effect_contribution(
    Y: pt.TensorVariable,
    psi_matrix: pt.TensorVariable,
    target_idx: int,
    n_outcomes: int,
    modulation: dict[int, pt.TensorVariable] | None = None,
) -> pt.TensorVariable:
    """
    Compute cross-effect contribution for a single target outcome.

    Parameters
    ----------
    Y : TensorVariable
        Outcome matrix (n_obs, n_outcomes)
    psi_matrix : TensorVariable
        Cross-effect coefficients (n_outcomes, n_outcomes)
    target_idx : int
        Index of target outcome
    n_outcomes : int
        Total number of outcomes
    modulation : dict | None
        Optional modulation by source index (e.g., promotion indicators)

    Returns
    -------
    TensorVariable
        Cross-effect contribution (n_obs,)
    """
    contribution = pt.zeros(Y.shape[0])

    for source_idx in range(n_outcomes):
        if source_idx == target_idx:
            continue

        psi = psi_matrix[source_idx, target_idx]

        if modulation and source_idx in modulation:
            # Modulated effect (e.g., only when source is promoted)
            effect = psi * modulation[source_idx]
        else:
            effect = psi

        contribution = contribution + effect * Y[:, source_idx]

    return contribution


# =============================================================================
# Regularized Horseshoe Prior
# =============================================================================


def create_regularized_horseshoe_prior(
    name: str,
    n_variables: int,
    n_obs: int,
    sigma: pt.TensorVariable,
    config: HorseshoeConfig,
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create regularized horseshoe prior (Piironen & Vehtari, 2017).

    The regularized horseshoe provides:
    - Strong shrinkage of small effects toward zero
    - Minimal shrinkage of large effects (they "escape" the horseshoe)
    - Slab regularization to prevent unrealistically large effects

    Model specification:
        beta_j = z_j * tau * lambda_tilde_j
        lambda_tilde_j = c * lambda_j / sqrt(c^2 + tau^2 * lambda_j^2)
        lambda_j ~ HalfStudentT(local_df)
        tau ~ HalfStudentT(global_df, scale=tau0)
        c^2 ~ InverseGamma(slab_df/2, slab_df * slab_scale^2 / 2)
        z_j ~ Normal(0, 1)

    where tau0 = D0/(D-D0) * sigma/sqrt(N) calibrates global shrinkage.

    Parameters
    ----------
    name : str
        Base name for the parameters.
    n_variables : int
        Number of variables (D).
    n_obs : int
        Number of observations (N).
    sigma : pt.TensorVariable
        Observation noise standard deviation.
    config : HorseshoeConfig
        Horseshoe configuration.
    dims : str | None
        PyMC dimension name for coefficients.

    Returns
    -------
    VariableSelectionResult
        Container with beta and diagnostic quantities.
    """
    D = n_variables
    D0 = min(config.expected_nonzero, D - 1)  # Ensure D0 < D
    N = n_obs

    # Global shrinkage scale (Piironen & Vehtari recommendation)
    # This calibration ensures prior expected number of nonzero ≈ D0
    tau0 = (D0 / (D - D0)) * (sigma / np.sqrt(N))

    # Global shrinkage parameter
    tau = pm.HalfStudentT(
        f"{name}_tau",
        nu=config.global_df,
        sigma=tau0,
    )

    # Local shrinkage parameters (one per variable)
    dim_kwargs = {"dims": dims} if dims else {"shape": D}
    lambda_local = pm.HalfStudentT(
        f"{name}_lambda",
        nu=config.local_df,
        **dim_kwargs,
    )

    # Slab regularization (c^2)
    # InverseGamma parameterized so E[c^2] ≈ slab_scale^2 when slab_df > 2
    c2 = pm.InverseGamma(
        f"{name}_c2",
        alpha=config.slab_df / 2,
        beta=config.slab_df * config.slab_scale**2 / 2,
    )

    # Regularized local shrinkage
    # lambda_tilde_j = c * lambda_j / sqrt(c^2 + tau^2 * lambda_j^2)
    # This bounds the effective scale by c, preventing very large coefficients
    lambda_tilde = pt.sqrt(c2) * lambda_local / pt.sqrt(c2 + tau**2 * lambda_local**2)

    # Standardized coefficients (non-centered parameterization)
    z = pm.Normal(f"{name}_z", mu=0, sigma=1, **dim_kwargs)

    # Final coefficients
    beta = pm.Deterministic(
        f"{name}",
        z * tau * lambda_tilde,
        dims=dims,
    )

    # Shrinkage factors kappa_j = 1 / (1 + tau^2 * lambda_j^2)
    # kappa near 1 = strong shrinkage, kappa near 0 = coefficient preserved
    kappa = pm.Deterministic(
        f"{name}_kappa",
        1 / (1 + tau**2 * lambda_local**2),
        dims=dims,
    )

    # Effective number of nonzero coefficients (diagnostic)
    effective_nonzero = pm.Deterministic(
        f"{name}_effective_nonzero",
        pt.sum(1 - kappa),
    )

    return VariableSelectionResult(
        beta=beta,
        local_shrinkage=lambda_local,
        global_shrinkage=tau,
        effective_nonzero=effective_nonzero,
        kappa=kappa,
    )


def create_finnish_horseshoe_prior(
    name: str,
    n_variables: int,
    n_obs: int,
    sigma: pt.TensorVariable,
    config: HorseshoeConfig,
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create Finnish horseshoe prior (Piironen & Vehtari, 2017).

    This is mathematically identical to the regularized horseshoe.
    The name emphasizes the slab regularization component.

    See create_regularized_horseshoe_prior for details.
    """
    return create_regularized_horseshoe_prior(
        name=name,
        n_variables=n_variables,
        n_obs=n_obs,
        sigma=sigma,
        config=config,
        dims=dims,
    )


# =============================================================================
# Spike-and-Slab Prior
# =============================================================================


def create_spike_slab_prior(
    name: str,
    n_variables: int,
    config: SpikeSlabConfig,
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create spike-and-slab prior for variable selection.

    The spike-and-slab uses a mixture of two distributions:
    - Spike: concentrated near zero (for "excluded" variables)
    - Slab: diffuse prior (for "included" variables)

    For gradient-based samplers (NUTS), we use a continuous relaxation
    where the discrete indicator is replaced by a continuous sigmoid.

    Model specification (continuous relaxation):
        beta_j = gamma_j * beta_slab_j + (1 - gamma_j) * beta_spike_j
        gamma_j = sigmoid(logit_gamma_j / temperature)
        logit_gamma_j ~ Normal(logit(pi), 1)
        beta_slab_j ~ Normal(0, slab_scale)
        beta_spike_j ~ Normal(0, spike_scale)

    Parameters
    ----------
    name : str
        Base name for the parameters.
    n_variables : int
        Number of variables.
    config : SpikeSlabConfig
        Spike-slab configuration.
    dims : str | None
        PyMC dimension name.

    Returns
    -------
    VariableSelectionResult
        Container with beta and inclusion indicators.
    """
    dim_kwargs = {"dims": dims} if dims else {"shape": n_variables}

    if config.use_continuous_relaxation:
        # Continuous relaxation for gradient-based sampling (NUTS)

        # Prior logit of inclusion probability
        prior_logit = np.log(
            config.prior_inclusion_prob / (1 - config.prior_inclusion_prob)
        )

        # Latent logit variable
        logit_gamma = pm.Normal(
            f"{name}_logit_gamma",
            mu=prior_logit,
            sigma=1.0,
            **dim_kwargs,
        )

        # Soft inclusion indicator via tempered sigmoid
        # Lower temperature = sharper selection (closer to discrete)
        gamma = pm.Deterministic(
            f"{name}_gamma",
            pm.math.sigmoid(logit_gamma / config.temperature),
            dims=dims,
        )

        # Slab component (nonzero effects)
        beta_slab = pm.Normal(
            f"{name}_slab",
            mu=0,
            sigma=config.slab_scale,
            **dim_kwargs,
        )

        # Spike component (near-zero effects)
        beta_spike = pm.Normal(
            f"{name}_spike",
            mu=0,
            sigma=config.spike_scale,
            **dim_kwargs,
        )

        # Mixture: interpolate between spike and slab
        beta = pm.Deterministic(
            f"{name}",
            gamma * beta_slab + (1 - gamma) * beta_spike,
            dims=dims,
        )

    else:
        # Discrete spike-and-slab (requires specialized sampler)
        # Warning: May have mixing issues with standard NUTS

        gamma = pm.Bernoulli(
            f"{name}_gamma",
            p=config.prior_inclusion_prob,
            **dim_kwargs,
        )

        beta_slab = pm.Normal(
            f"{name}_slab",
            mu=0,
            sigma=config.slab_scale,
            **dim_kwargs,
        )

        # Beta is zero when gamma=0, slab value when gamma=1
        beta = pm.Deterministic(
            f"{name}",
            gamma * beta_slab,
            dims=dims,
        )

    # Effective count of included variables
    effective_nonzero = pm.Deterministic(
        f"{name}_effective_nonzero",
        pt.sum(gamma),
    )

    return VariableSelectionResult(
        beta=beta,
        inclusion_indicators=gamma,
        effective_nonzero=effective_nonzero,
    )


# =============================================================================
# Bayesian LASSO Prior
# =============================================================================


def create_bayesian_lasso_prior(
    name: str,
    n_variables: int,
    config: LassoConfig,
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create Bayesian LASSO prior (Park & Casella, 2008).

    The Bayesian LASSO places a Laplace (double exponential) prior on
    coefficients, represented as a scale mixture of normals.

    Model specification:
        beta_j | tau_j ~ Normal(0, sqrt(tau_j))
        tau_j ~ Exponential(lambda^2 / 2)

    This is equivalent to: beta_j ~ Laplace(0, 1/lambda)

    Parameters
    ----------
    name : str
        Base name for the parameters.
    n_variables : int
        Number of variables.
    config : LassoConfig
        LASSO configuration.
    dims : str | None
        PyMC dimension name.

    Returns
    -------
    VariableSelectionResult
        Container with beta and scale parameters.
    """
    dim_kwargs = {"dims": dims} if dims else {"shape": n_variables}

    # Scale mixture representation of Laplace
    # tau_j ~ Exponential(lambda^2 / 2)
    tau = pm.Exponential(
        f"{name}_tau",
        lam=config.regularization**2 / 2,
        **dim_kwargs,
    )

    # beta_j | tau_j ~ Normal(0, sqrt(tau_j))
    beta = pm.Normal(
        f"{name}",
        mu=0,
        sigma=pt.sqrt(tau),
        **dim_kwargs,
    )

    return VariableSelectionResult(
        beta=beta,
        local_shrinkage=tau,
    )


# =============================================================================
# Main Factory Function
# =============================================================================


def create_variable_selection_prior(
    name: str,
    n_variables: int,
    n_obs: int,
    sigma: pt.TensorVariable,
    config: VariableSelectionConfig,
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Factory function to create variable selection priors.

    This is the main entry point for adding variable selection to a model.
    It dispatches to the appropriate prior implementation based on the
    configuration method.

    CAUSAL WARNING: This should only be used for precision control variables,
    not confounders. Use config.exclude_variables to ensure confounders
    are handled separately with standard priors.

    Parameters
    ----------
    name : str
        Base name for the coefficient parameters.
    n_variables : int
        Number of control variables subject to selection.
    n_obs : int
        Number of observations.
    sigma : pt.TensorVariable
        Observation noise standard deviation.
    config : VariableSelectionConfig
        Complete configuration specifying method and hyperparameters.
    dims : str | None
        PyMC dimension name for the coefficient vector.

    Returns
    -------
    VariableSelectionResult
        Container with coefficient vector and diagnostic quantities.

    Examples
    --------
    >>> config = VariableSelectionConfig(
    ...     method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
    ...     horseshoe=HorseshoeConfig(expected_nonzero=3),
    ... )
    >>> with pm.Model(coords={"control": control_names}) as model:
    ...     sigma = pm.HalfNormal("sigma", 1)
    ...     result = create_variable_selection_prior(
    ...         "beta_controls",
    ...         n_variables=len(control_names),
    ...         n_obs=156,
    ...         sigma=sigma,
    ...         config=config,
    ...         dims="control",
    ...     )
    ...     # Use result.beta in likelihood
    """
    method = config.method

    if method == VariableSelectionMethod.NONE:
        # Standard normal priors (no selection)
        dim_kwargs = {"dims": dims} if dims else {"shape": n_variables}
        beta = pm.Normal(
            name,
            mu=0,
            sigma=0.5,
            **dim_kwargs,
        )
        return VariableSelectionResult(beta=beta)

    elif method == VariableSelectionMethod.REGULARIZED_HORSESHOE:
        return create_regularized_horseshoe_prior(
            name=name,
            n_variables=n_variables,
            n_obs=n_obs,
            sigma=sigma,
            config=config.horseshoe,
            dims=dims,
        )

    elif method == VariableSelectionMethod.FINNISH_HORSESHOE:
        return create_finnish_horseshoe_prior(
            name=name,
            n_variables=n_variables,
            n_obs=n_obs,
            sigma=sigma,
            config=config.horseshoe,
            dims=dims,
        )

    elif method == VariableSelectionMethod.SPIKE_SLAB:
        return create_spike_slab_prior(
            name=name,
            n_variables=n_variables,
            config=config.spike_slab,
            dims=dims,
        )

    elif method == VariableSelectionMethod.BAYESIAN_LASSO:
        return create_bayesian_lasso_prior(
            name=name,
            n_variables=n_variables,
            config=config.lasso,
            dims=dims,
        )

    else:
        raise ValueError(f"Unknown variable selection method: {method}")


# =============================================================================
# Control Effect Builder with Selection
# =============================================================================


@dataclass
class ControlEffectResult:
    """
    Container for control variable effects with optional selection.

    Attributes
    ----------
    contribution : pt.TensorVariable
        Total control contribution (n_obs,).
    beta_selected : pt.TensorVariable | None
        Coefficients for selected (shrinkage) variables.
    beta_fixed : pt.TensorVariable | None
        Coefficients for fixed (non-shrinkage) variables.
    selection_result : VariableSelectionResult | None
        Full selection result for diagnostics.
    components : dict[str, pt.TensorVariable]
        Individual variable contributions.
    """

    contribution: pt.TensorVariable
    beta_selected: pt.TensorVariable | None = None
    beta_fixed: pt.TensorVariable | None = None
    selection_result: VariableSelectionResult | None = None
    components: dict[str, pt.TensorVariable] = field(default_factory=dict)


def build_control_effects_with_selection(
    X_controls: np.ndarray | pt.TensorVariable,
    control_names: list[str],
    n_obs: int,
    sigma: pt.TensorVariable,
    selection_config: VariableSelectionConfig,
    name_prefix: str = "control",
) -> ControlEffectResult:
    """
    Build control variable effects with optional variable selection.

    This function handles the split between variables subject to selection
    (precision controls) and those excluded from selection (confounders).

    Parameters
    ----------
    X_controls : array-like
        Control variable matrix (n_obs, n_controls).
    control_names : list[str]
        Names of control variables.
    n_obs : int
        Number of observations.
    sigma : pt.TensorVariable
        Observation noise (for horseshoe calibration).
    selection_config : VariableSelectionConfig
        Configuration for variable selection.
    name_prefix : str
        Prefix for parameter names.

    Returns
    -------
    ControlEffectResult
        Container with contributions and coefficients.

    Examples
    --------
    >>> config = VariableSelectionConfig(
    ...     method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
    ...     horseshoe=HorseshoeConfig(expected_nonzero=3),
    ...     exclude_variables=("distribution", "price"),
    ... )
    >>> result = build_control_effects_with_selection(
    ...     X_controls, control_names, n_obs, sigma, config
    ... )
    >>> # result.contribution is the total control effect
    """
    X_controls = pt.as_tensor_variable(X_controls)

    # Partition variables into selectable and fixed
    selectable, fixed = selection_config.get_selectable_variables(control_names)

    components = {}
    contribution_parts = []
    beta_selected = None
    beta_fixed = None
    selection_result = None

    # Handle fixed (non-shrinkage) variables - typically confounders
    if fixed:
        fixed_idx = [control_names.index(v) for v in fixed]
        X_fixed = X_controls[:, fixed_idx]

        # Standard normal priors for fixed controls
        beta_fixed = pm.Normal(
            f"{name_prefix}_fixed",
            mu=0,
            sigma=0.5,
            dims=f"{name_prefix}_fixed_dim" if len(fixed) > 1 else None,
            shape=len(fixed) if len(fixed) > 1 else (),
        )

        if len(fixed) == 1:
            fixed_contrib = beta_fixed * X_fixed[:, 0]
            components[fixed[0]] = fixed_contrib
        else:
            fixed_contrib = pt.dot(X_fixed, beta_fixed)
            for i, var_name in enumerate(fixed):
                components[var_name] = (
                    beta_fixed[i] * X_controls[:, control_names.index(var_name)]
                )

        contribution_parts.append(fixed_contrib)

    # Handle selectable (shrinkage) variables - precision controls
    if selectable and selection_config.method != VariableSelectionMethod.NONE:
        selectable_idx = [control_names.index(v) for v in selectable]
        X_selectable = X_controls[:, selectable_idx]

        # Create selection prior
        selection_result = create_variable_selection_prior(
            name=f"{name_prefix}_select",
            n_variables=len(selectable),
            n_obs=n_obs,
            sigma=sigma,
            config=selection_config,
            dims=f"{name_prefix}_select_dim",
        )
        beta_selected = selection_result.beta

        selectable_contrib = pt.dot(X_selectable, beta_selected)
        contribution_parts.append(selectable_contrib)

        # Store individual components
        for i, var_name in enumerate(selectable):
            components[var_name] = (
                beta_selected[i] * X_controls[:, control_names.index(var_name)]
            )

    elif selectable:
        # No selection - standard priors for all selectable
        selectable_idx = [control_names.index(v) for v in selectable]
        X_selectable = X_controls[:, selectable_idx]

        beta_selected = pm.Normal(
            f"{name_prefix}_select",
            mu=0,
            sigma=0.5,
            shape=len(selectable),
        )

        selectable_contrib = pt.dot(X_selectable, beta_selected)
        contribution_parts.append(selectable_contrib)

        for i, var_name in enumerate(selectable):
            components[var_name] = (
                beta_selected[i] * X_controls[:, control_names.index(var_name)]
            )

    # Combine contributions
    if contribution_parts:
        total_contribution = sum(contribution_parts)
    else:
        total_contribution = pt.zeros(n_obs)

    return ControlEffectResult(
        contribution=total_contribution,
        beta_selected=beta_selected,
        beta_fixed=beta_fixed,
        selection_result=selection_result,
        components=components,
    )


# =============================================================================
# Diagnostic Utilities
# =============================================================================


def compute_inclusion_probabilities(
    trace: "az.InferenceData",
    config: VariableSelectionConfig,
    name: str = "beta_controls",
    threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Compute posterior inclusion probabilities from fitted model.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples from fitted model.
    config : VariableSelectionConfig
        Configuration used for fitting.
    name : str
        Base name of the coefficient parameters.
    threshold : float
        For horseshoe: signal-to-noise threshold for "inclusion".

    Returns
    -------
    dict
        Dictionary with 'inclusion_prob' array and 'effective_nonzero'.
    """
    posterior = trace.posterior

    if config.method == VariableSelectionMethod.SPIKE_SLAB:
        # Direct inclusion indicators available
        gamma_name = f"{name}_gamma"
        if gamma_name in posterior:
            gamma_samples = posterior[gamma_name].values
            inclusion_prob = gamma_samples.mean(axis=(0, 1))
            effective_nonzero = inclusion_prob.sum()
        else:
            raise ValueError(f"Could not find {gamma_name} in posterior")

    elif config.method in [
        VariableSelectionMethod.REGULARIZED_HORSESHOE,
        VariableSelectionMethod.FINNISH_HORSESHOE,
    ]:
        # Use shrinkage factors or magnitude-based heuristic
        kappa_name = f"{name}_kappa"
        if kappa_name in posterior:
            # 1 - kappa gives "inclusion strength"
            kappa_samples = posterior[kappa_name].values
            inclusion_prob = 1 - kappa_samples.mean(axis=(0, 1))
            effective_nonzero = inclusion_prob.sum()
        else:
            # Fall back to magnitude-based
            beta_samples = posterior[name].values
            beta_mean = np.abs(beta_samples.mean(axis=(0, 1)))
            beta_std = beta_samples.std(axis=(0, 1)) + 1e-10
            snr = beta_mean / beta_std
            inclusion_prob = (snr > threshold).astype(float)
            effective_nonzero = inclusion_prob.sum()

    else:
        # For other methods, use credible interval heuristic
        beta_samples = posterior[name].values
        lower = np.percentile(beta_samples, 2.5, axis=(0, 1))
        upper = np.percentile(beta_samples, 97.5, axis=(0, 1))
        inclusion_prob = ((lower > 0) | (upper < 0)).astype(float)
        effective_nonzero = inclusion_prob.sum()

    return {
        "inclusion_prob": inclusion_prob,
        "effective_nonzero": effective_nonzero,
    }


def summarize_variable_selection(
    trace: "az.InferenceData",
    control_names: list[str],
    config: VariableSelectionConfig,
    name: str = "beta_controls",
) -> "pd.DataFrame":
    """
    Create summary table of variable selection results.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples.
    control_names : list[str]
        Names of control variables.
    config : VariableSelectionConfig
        Configuration used.
    name : str
        Base parameter name.

    Returns
    -------
    pd.DataFrame
        Summary with columns: variable, mean, std, hdi_3%, hdi_97%,
        inclusion_prob, selected.
    """
    import pandas as pd

    posterior = trace.posterior
    beta_samples = posterior[name].values

    # Get inclusion probabilities
    inclusion_info = compute_inclusion_probabilities(trace, config, name)

    # Build summary
    summary_data = []
    for i, var_name in enumerate(control_names):
        var_samples = beta_samples[:, :, i].flatten()
        summary_data.append(
            {
                "variable": var_name,
                "mean": var_samples.mean(),
                "std": var_samples.std(),
                "hdi_3%": np.percentile(var_samples, 3),
                "hdi_97%": np.percentile(var_samples, 97),
                "inclusion_prob": inclusion_info["inclusion_prob"][i],
                "selected": inclusion_info["inclusion_prob"][i] > 0.5,
            }
        )

    df = pd.DataFrame(summary_data)
    df = df.sort_values("inclusion_prob", ascending=False)

    return df
