"""
Prior factory functions for MMM Extensions.

These functions create PyMC prior distributions for various
model parameters with appropriate configurations.
"""

from __future__ import annotations

import pymc as pm
import pytensor.tensor as pt


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


__all__ = [
    "create_adstock_prior",
    "create_saturation_prior",
    "create_effect_prior",
]
