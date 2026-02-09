"""
Model component builders for MMM Extensions.

These functions build reusable model components (media transforms,
linear effects) that can be composed into larger models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytensor.tensor as pt

from .transforms import (
    geometric_adstock_convolution,
    logistic_saturation_pt,
    hill_saturation,
)
from .priors import create_adstock_prior, create_saturation_prior, create_effect_prior


@dataclass
class MediaTransformResult:
    """Result of media transformation."""

    transformed: pt.TensorVariable  # (n_obs, n_channels)
    adstock_params: dict[str, pt.TensorVariable]
    saturation_params: dict[str, pt.TensorVariable]


@dataclass
class EffectResult:
    """Result of effect computation."""

    contribution: pt.TensorVariable  # (n_obs,)
    coefficients: pt.TensorVariable
    components: pt.TensorVariable | None = None  # (n_obs, n_vars) if multiple


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
        x_adstocked = geometric_adstock_convolution(x, alpha, l_max)

        # Saturation
        sat_type = saturation_config.get("type", "logistic")
        sat_params = create_saturation_prior(
            f"{prefix}sat_{channel}",
            saturation_type=sat_type,
            **saturation_config.get("prior_params", {}),
        )
        saturation_params[channel] = sat_params

        if sat_type == "logistic":
            x_saturated = logistic_saturation_pt(x_adstocked, sat_params["lam"])
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


__all__ = [
    "MediaTransformResult",
    "EffectResult",
    "build_media_transforms",
    "build_linear_effect",
]
