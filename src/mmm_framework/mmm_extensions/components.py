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
from typing import Protocol, Callable, Any
from enum import Enum


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
    result = np.convolve(x, weights[::-1], mode='full')[:len(x)]
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
        window = x_padded[i:i + l_max]
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
        return {
            "lam": pm.Gamma(f"{name}_lam", alpha=lam_alpha, beta=lam_beta)
        }
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
                x_adstocked, 
                sat_params["kappa"], 
                sat_params["slope"]
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
        
        psi_matrix = pt.set_subtensor(
            psi_matrix[spec.source_idx, spec.target_idx],
            psi
        )
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