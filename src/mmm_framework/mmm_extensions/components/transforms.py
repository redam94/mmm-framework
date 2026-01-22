"""
Extension-specific transformation functions.

Most transforms are re-exported from mmm_framework.transforms.
This module contains only extensions-specific implementations
that don't exist in the base module.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytensor.tensor as pt
from pytensor import scan as pytensor_scan


def geometric_adstock_pt(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    l_max: int = 8,
    normalize: bool = True,
) -> pt.TensorVariable:
    """
    Apply geometric adstock transformation using PyTensor scan.

    This version uses scan for proper gradient flow in complex models.
    For most use cases, geometric_adstock_convolution is preferred.

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

    # Build convolution using indexing
    n = x.shape[0]
    indices = pt.arange(l_max)

    # Build a matrix where each row is a window
    row_indices = pt.arange(n)[:, None] + indices[None, :]
    windows = x_padded[row_indices]  # (n, l_max)

    return pt.dot(windows, weights[::-1])


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
        results.append(geometric_adstock_convolution(X[:, i], alphas[i], l_max))
    return pt.stack(results, axis=1)


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


def logistic_saturation_pt(
    x: pt.TensorVariable,
    lam: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Apply logistic saturation transformation (PyTensor version).

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
    Apply Hill saturation transformation (PyTensor version).

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


__all__ = [
    "geometric_adstock_pt",
    "geometric_adstock_convolution",
    "geometric_adstock_matrix",
    "apply_transformation_pipeline",
    "logistic_saturation_pt",
    "hill_saturation",
]
