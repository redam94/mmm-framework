"""Graph-faithful numpy mirrors of the model's media transforms.

The frequentist estimation path (epic #180) builds a design matrix **out of** the
PyTensor graph: with each channel's adstock and saturation parameters held fixed,
the model is linear in everything else, so the transformed media series become
constant columns and the fit is a linear solve. That only works if the numpy
transforms reproduce the graph's arithmetic *exactly* — if the two drift, a
frequentist fit and a Bayesian fit stop being comparable and every benchmark
between them is meaningless.

Why this module exists rather than reusing :mod:`mmm_framework.transforms`
-------------------------------------------------------------------------
:func:`mmm_framework.transforms.adstock.parametric_adstock` **is** faithful (it
agrees with the in-graph kernel to ~1e-12) and is reused here directly.

The saturation twins in :mod:`mmm_framework.transforms.saturation` are **not**:

* ``logistic_saturation`` computes ``1 - exp(-lam * clip(x, 0, None))`` while the
  graph clips the *exponent* (``pt.clip(-sat_lam * x, -20, 0)``). They agree until
  ``lam * x > 20`` and then diverge by ~2e-9 — and the transform search ranges over
  ``lam`` on media normalized to roughly ``[0, 1]``, so that region is reachable.
* ``root_saturation`` computes ``clip(x, 0, None) ** k`` while the graph clamps
  ``pt.maximum(x, 1e-9)`` before the power. At a zero-spend row they differ by
  ``1e-9 ** 0.5 = 3.16e-05`` — five orders of magnitude above the tolerance the
  equivalence test needs, on every flighted week.
* ``hill``, ``michaelis_menten`` and ``tanh`` have no numpy twin there at all.

Those helpers are public API with their own documented semantics, so they are left
alone; this module is the mirror of the *graph*, and
:func:`mmm_framework.model.base._apply_saturation_pt` is the single thing it
tracks. Any change there must be mirrored here, and
``tests/frequentist/test_saturation_fidelity.py`` fails if it is not.

:class:`~mmm_framework.validation.backtest.PosteriorForecaster` also needs these
numpy forms and previously carried its own copy, which silently omitted
``SaturationType.ROOT`` — a root-saturation channel was forecast *unsaturated*.
It now imports :func:`saturate`, so there is one definition and one test guarding
both callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.enums import SaturationType
from ..transforms.adstock import adstock_weights, apply_adstock

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["saturate", "adstock_series", "adstock_panel", "SATURATION_PARAMS"]


#: Parameter names each saturation family consumes, matching the keys
#: ``BayesianMMM._build_channel_saturation`` puts in its params dict.
SATURATION_PARAMS: dict[SaturationType, tuple[str, ...]] = {
    SaturationType.LOGISTIC: ("sat_lam",),
    SaturationType.HILL: ("sat_half", "sat_slope"),
    SaturationType.MICHAELIS_MENTEN: ("sat_half",),
    SaturationType.TANH: ("sat_half",),
    SaturationType.ROOT: ("sat_exponent",),
    SaturationType.NONE: (),
}


def saturate(
    x: "NDArray[np.floating]",
    kind: SaturationType,
    **params: Any,
) -> "NDArray[np.floating]":
    """Apply a channel's saturation in numpy, mirroring the in-graph form.

    This is the numpy twin of
    :func:`mmm_framework.model.base._apply_saturation_pt`, **including its
    numerical guards** — the exponent clip on the logistic form and the
    ``maximum(x, 1e-9)`` clamps on the root and Hill forms. Those guards are not
    cosmetic: they are what the fitted graph actually computes, so dropping them
    makes the design matrix disagree with the likelihood.

    Args:
        x: Adstocked, normalized (roughly ``[0, 1]``) media. Any shape.
        kind: The channel's configured :class:`SaturationType`.
        **params: The family's parameters (see :data:`SATURATION_PARAMS`), each
            either a scalar or an array broadcastable against ``x``. Callers that
            evaluate many draws at once — e.g.
            :class:`~mmm_framework.validation.backtest.PosteriorForecaster`, which
            passes ``x`` of shape ``(n_obs, n_draws)`` — are responsible for
            shaping their parameters to broadcast (``lam[None, :]``).

    Returns:
        The saturated array, same shape as the broadcast of ``x`` and ``params``.

    Raises:
        KeyError: If a parameter the family requires was not supplied.
        ValueError: If ``kind`` is not a recognized :class:`SaturationType`.
    """
    if kind == SaturationType.LOGISTIC:
        # Graph: 1 - exp(clip(-sat_lam * x, -20, 0)). Clipping the EXPONENT (not
        # x) is what bounds the result away from exactly 1.
        return 1.0 - np.exp(np.clip(-params["sat_lam"] * x, -20, 0))

    if kind == SaturationType.ROOT:
        # Graph clamps x away from 0 before the power: d/dx x^k is unbounded at
        # x = 0 for k < 1, which would hand NUTS infinite gradients on zero-spend
        # weeks. The clamp therefore shows up in the fitted values.
        return np.maximum(x, 1e-9) ** params["sat_exponent"]

    if kind == SaturationType.HILL:
        x_pow = np.maximum(x, 1e-9) ** params["sat_slope"]
        return x_pow / (x_pow + params["sat_half"] ** params["sat_slope"])

    if kind == SaturationType.MICHAELIS_MENTEN:
        return x / (x + params["sat_half"])

    if kind == SaturationType.TANH:
        return np.tanh(x / params["sat_half"])

    if kind == SaturationType.NONE:
        return np.asarray(x)

    raise ValueError(f"Unknown saturation type: {kind!r}")


def adstock_series(
    x_norm: "NDArray[np.floating]",
    kind: str,
    l_max: int,
    *,
    normalize: bool = True,
    **params: float,
) -> "NDArray[np.floating]":
    """Adstock a single contiguous, time-ordered series.

    Thin wrapper over :func:`~mmm_framework.transforms.adstock.adstock_weights`
    plus :func:`~mmm_framework.transforms.adstock.apply_adstock` — the same pair
    the in-graph kernel mirrors, so no separate fidelity argument is needed.

    Args:
        x_norm: Normalized media for one channel and one cell, time-ordered.
        kind: ``"geometric"``, ``"delayed"``, ``"weibull"`` or ``"none"``.
        l_max: Kernel length in periods.
        normalize: Whether the kernel sums to 1 (magnitude absorbed into beta).
        **params: ``alpha`` / ``theta`` / ``shape`` / ``scale`` per the family.

    Returns:
        The adstocked series, same length as ``x_norm``.
    """
    if kind == "none":
        return np.asarray(x_norm, dtype=float)
    weights = adstock_weights(kind, l_max, normalize=normalize, **params)
    return apply_adstock(x_norm, weights)


def adstock_panel(
    x_norm: "NDArray[np.floating]",
    kind: str,
    l_max: int,
    *,
    time_idx: "NDArray[np.integer]",
    cell_idx: "NDArray[np.integer]",
    n_cells: int,
    normalize: bool = True,
    **params: float,
) -> "NDArray[np.floating]":
    """Adstock a panel series **per cell**, mirroring the in-graph panel kernel.

    A panel's observations are stacked, not contiguous in time within a cell, so
    convolving the flat vector lets one geography's spend carry over into the
    next geography's first weeks. On a 3-geo panel that bleed is worth ~0.26 in
    normalized units — far larger than any real effect this path estimates. The
    graph avoids it with :func:`parametric_adstock_panel_pt`; this is its numpy
    twin, and it agrees to ~3e-13.

    Args:
        x_norm: Normalized media for one channel, one entry per observation.
        kind: Kernel family, as :func:`adstock_series`.
        l_max: Kernel length in periods.
        time_idx: Period index per observation.
        cell_idx: Cell (geo × product) index per observation.
        n_cells: Number of cells.
        normalize: Whether the kernel sums to 1.
        **params: Kernel parameters per the family.

    Returns:
        The adstocked series, aligned to the input's observation order.
    """
    x_norm = np.asarray(x_norm, dtype=float)
    if kind == "none":
        return x_norm.copy()
    if n_cells <= 1:
        return adstock_series(x_norm, kind, l_max, normalize=normalize, **params)

    out = np.empty_like(x_norm)
    for cell in range(n_cells):
        rows = np.flatnonzero(cell_idx == cell)
        if rows.size == 0:
            continue
        # Sort by period so the convolution sees the cell's own history in order,
        # then scatter back to the caller's row order.
        order = np.argsort(time_idx[rows], kind="stable")
        ordered = rows[order]
        out[ordered] = adstock_series(
            x_norm[ordered], kind, l_max, normalize=normalize, **params
        )
    return out
