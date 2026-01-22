"""
Cross-effect builders for multivariate MMM.

These functions build cross-effect structures (cannibalization,
halo effects) between outcomes in multivariate models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymc as pm
import pytensor.tensor as pt


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


__all__ = [
    "CrossEffectSpec",
    "build_cross_effect_matrix",
    "compute_cross_effect_contribution",
]
