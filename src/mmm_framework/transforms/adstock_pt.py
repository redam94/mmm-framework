"""PyTensor (in-graph) adstock kernels for Bayesian estimation.

These mirror the NumPy reference in :mod:`mmm_framework.transforms.adstock`
exactly, so a model that estimates adstock parameters in-graph produces the
same kernel as the NumPy helpers used for analysis and reporting.

This module is intentionally separate from ``transforms/__init__`` so importing
``mmm_framework.transforms`` stays free of the PyTensor dependency; import these
functions directly from ``mmm_framework.transforms.adstock_pt`` where needed.

All kernels are finite-impulse-response (FIR) convolutions differing only in the
lag-indexed weight vector. ``w[0]`` weights the current period, so geometric
weights peak at lag 0 while delayed/Weibull weights can peak later.
"""

from __future__ import annotations

import pytensor.tensor as pt


def adstock_weights_pt(
    kind: str,
    l_max: int,
    *,
    alpha: pt.TensorVariable | float = 0.5,
    theta: pt.TensorVariable | float = 0.0,
    shape: pt.TensorVariable | float = 2.0,
    scale: pt.TensorVariable | float = 2.0,
    normalize: bool = True,
) -> pt.TensorVariable:
    """Build a lag-indexed FIR adstock weight vector (PyTensor).

    Parameters
    ----------
    kind : {"geometric", "delayed", "weibull", "none"}
        Kernel shape.
    l_max : int
        Kernel length (number of lags).
    alpha : TensorVariable or float
        Decay rate for ``"geometric"`` / ``"delayed"``.
    theta : TensorVariable or float
        Peak/delay lag for ``"delayed"`` (0 reproduces geometric).
    shape, scale : TensorVariable or float
        Weibull shape ``k`` and scale ``lambda``.
    normalize : bool
        If True, weights are scaled to sum to 1.
    """
    lags = pt.arange(l_max)

    if kind == "none":
        return pt.concatenate([pt.ones(1), pt.zeros(l_max - 1)])
    if kind == "geometric":
        w = pt.power(alpha, lags)
    elif kind == "delayed":
        w = pt.power(alpha, (lags - theta) ** 2)
    elif kind == "weibull":
        # Weibull PDF at lags shifted by 1 so lag 0 is finite for every shape.
        t = lags + 1.0
        w = (
            (shape / scale)
            * (t / scale) ** (shape - 1.0)
            * pt.exp(-((t / scale) ** shape))
        )
    else:
        raise ValueError(f"Unknown adstock kind: {kind!r}")

    if normalize:
        # Epsilon guards the 0/0 case: extreme parameter draws (e.g. Weibull
        # with large shape, or delayed with alpha ~ 0 and fractional theta) can
        # underflow every weight to 0, and a NaN here poisons the whole graph.
        w = w / (w.sum() + 1e-12)
    return w


def apply_adstock_pt(
    x: pt.TensorVariable,
    weights: pt.TensorVariable,
    l_max: int,
) -> pt.TensorVariable:
    """Convolve a series with an FIR adstock kernel (causal, no scan).

    Computes ``y[t] = sum_k weights[k] * x[t - k]`` with causal zero padding,
    matching :func:`mmm_framework.transforms.adstock.apply_adstock`.
    """
    x_padded = pt.concatenate([pt.zeros(l_max - 1), x])
    n = x.shape[0]
    row_indices = pt.arange(n)[:, None] + pt.arange(l_max)[None, :]
    windows = x_padded[row_indices]  # (n, l_max)
    return pt.dot(windows, weights[::-1])


def apply_adstock_panel_pt(
    x: pt.TensorVariable,
    weights: pt.TensorVariable,
    l_max: int,
    *,
    time_idx,
    cell_idx,
    n_periods: int,
    n_cells: int,
) -> pt.TensorVariable:
    """Convolve a stacked panel series with an FIR kernel, per cross-section.

    A panel model stacks ``n_periods x n_cells`` observations into one vector
    (a "cell" is a geography, a product, or a geography x product combination).
    Convolving that stacked vector directly would let carryover bleed across
    cell boundaries — one geography's spend would adstock into *another
    geography's* following rows. This scatters the observations into a
    ``(n_periods, n_cells)`` matrix using each observation's time/cell index,
    convolves along the time axis only, and gathers back to the original
    stacked layout, so every cell carries over only its own history.

    Missing (time, cell) combinations in an unbalanced panel contribute zero
    spend, which is the correct causal treatment for weeks with no recorded
    activity.
    """
    x_mat = pt.zeros((n_periods, n_cells))
    x_mat = pt.set_subtensor(x_mat[time_idx, cell_idx], x)
    x_padded = pt.concatenate([pt.zeros((l_max - 1, n_cells)), x_mat], axis=0)
    row_indices = pt.arange(n_periods)[:, None] + pt.arange(l_max)[None, :]
    windows = x_padded[row_indices]  # (n_periods, l_max, n_cells)
    y_mat = pt.tensordot(windows, weights[::-1], axes=[[1], [0]])
    return y_mat[time_idx, cell_idx]


def parametric_adstock_pt(
    x: pt.TensorVariable,
    kind: str,
    l_max: int = 8,
    *,
    alpha: pt.TensorVariable | float = 0.5,
    theta: pt.TensorVariable | float = 0.0,
    shape: pt.TensorVariable | float = 2.0,
    scale: pt.TensorVariable | float = 2.0,
    normalize: bool = True,
) -> pt.TensorVariable:
    """Apply geometric/delayed/Weibull FIR adstock to a 1D series (PyTensor).

    Dispatches on ``kind`` to build the kernel, then convolves. This is the
    in-graph counterpart used when adstock parameters are estimated.
    """
    weights = adstock_weights_pt(
        kind,
        l_max,
        alpha=alpha,
        theta=theta,
        shape=shape,
        scale=scale,
        normalize=normalize,
    )
    return apply_adstock_pt(x, weights, l_max)


def parametric_adstock_panel_pt(
    x: pt.TensorVariable,
    kind: str,
    l_max: int = 8,
    *,
    time_idx,
    cell_idx,
    n_periods: int,
    n_cells: int,
    alpha: pt.TensorVariable | float = 0.5,
    theta: pt.TensorVariable | float = 0.0,
    shape: pt.TensorVariable | float = 2.0,
    scale: pt.TensorVariable | float = 2.0,
    normalize: bool = True,
) -> pt.TensorVariable:
    """Panel-aware :func:`parametric_adstock_pt` (per geography/product cell).

    Same kernel families, but the convolution runs along each cross-section
    cell's own time axis (see :func:`apply_adstock_panel_pt`), so carryover
    never crosses a geography/product boundary.
    """
    weights = adstock_weights_pt(
        kind,
        l_max,
        alpha=alpha,
        theta=theta,
        shape=shape,
        scale=scale,
        normalize=normalize,
    )
    return apply_adstock_panel_pt(
        x,
        weights,
        l_max,
        time_idx=time_idx,
        cell_idx=cell_idx,
        n_periods=n_periods,
        n_cells=n_cells,
    )
