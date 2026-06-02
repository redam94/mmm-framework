"""
Model component builders for MMM Extensions.

These functions build reusable model components (media transforms,
linear effects) that can be composed into larger models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pymc as pm
import pytensor.tensor as pt

from .transforms import (
    logistic_saturation_pt,
    hill_saturation,
    parametric_adstock_pt,
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
    prefix = f"{name_prefix}_" if name_prefix else ""

    adstock_params = {}
    saturation_params = {}
    transformed_channels = []

    adstock_type = adstock_config.get("type", "geometric")
    l_max = adstock_config.get("l_max", 8)
    normalize = adstock_config.get("normalize", True)

    # Create the shared decay prior only for kernels that use alpha, so we
    # don't introduce an unused (orphan) RV for Weibull/none.
    if share_params and adstock_type in ("geometric", "delayed"):
        alpha = create_adstock_prior(
            f"{prefix}alpha_shared",
            prior_type=adstock_config.get("prior_type", "beta"),
            **adstock_config.get("prior_params", {}),
        )
        adstock_params["shared"] = alpha

    # Transform each channel
    for i, channel in enumerate(channel_names):
        x = X_media[:, i]

        # Adstock — dispatch on kernel shape so delayed/Weibull are honored
        # rather than silently falling back to geometric.
        if adstock_type == "none":
            x_adstocked = x
        elif adstock_type in ("geometric", "delayed"):
            if share_params:
                alpha = adstock_params["shared"]
            else:
                alpha = create_adstock_prior(
                    f"{prefix}alpha_{channel}",
                    prior_type=adstock_config.get("prior_type", "beta"),
                    **adstock_config.get("prior_params", {}),
                )
                adstock_params[channel] = alpha

            if adstock_type == "geometric":
                x_adstocked = parametric_adstock_pt(
                    x, "geometric", l_max, alpha=alpha, normalize=normalize
                )
            else:
                theta = pm.HalfNormal(
                    f"{prefix}theta_{channel}",
                    sigma=adstock_config.get("theta_sigma", 2.0),
                )
                adstock_params[f"{channel}_theta"] = theta
                x_adstocked = parametric_adstock_pt(
                    x, "delayed", l_max, alpha=alpha, theta=theta, normalize=normalize
                )
        elif adstock_type == "weibull":
            shape = pm.Gamma(
                f"{prefix}shape_{channel}",
                alpha=adstock_config.get("shape_alpha", 2.0),
                beta=adstock_config.get("shape_beta", 1.0),
            )
            scale = pm.Gamma(
                f"{prefix}scale_{channel}",
                alpha=adstock_config.get("scale_alpha", 2.0),
                beta=adstock_config.get("scale_beta", 1.0),
            )
            adstock_params[channel] = shape
            adstock_params[f"{channel}_scale"] = scale
            x_adstocked = parametric_adstock_pt(
                x, "weibull", l_max, shape=shape, scale=scale, normalize=normalize
            )
        else:
            raise ValueError(f"Unknown adstock type: {adstock_type!r}")

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
