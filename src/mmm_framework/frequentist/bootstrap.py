"""Bootstrap intervals for the frequentist path — the honest way to make draws.

A ridge fit is a point. Everything downstream in this framework consumes draws:
estimand CIs, ROI forests, report tables, the interactive report's recompute
engine. This module produces them, and the whole design is organized around the
three ways that goes wrong.

**1. Autocorrelation.** MMM residuals are serially correlated. An iid residual
bootstrap treats each week as exchangeable, understates the variance and returns
intervals that are too narrow — the same error class as the AR(1) design-effect
correction in the docs work and the autocorrelation-inflated false-positive rate
in ``planning/simulation.py``. So the resampling unit is a **moving block** of
consecutive periods, and the block length is estimated from the residual
autocorrelation rather than assumed. Measured over 60 simulations × 300
replicates of the ``make_clean`` world with AR(1) errors, per-channel total
contributions at the 90% level:

===========  ====================  ===================================
ρ            iid bootstrap         block bootstrap
===========  ====================  ===================================
0.6          **79.6%** (70–93)     **90.4%** (82–97), median block 7
0.0          92.9% (83–97)         93.3% (85–97), median block 1
===========  ====================  ===================================

At ρ = 0 the estimated block length collapses to 1 and the two agree, so nothing
pays a width penalty for a dependence that is not there. The one channel still
below nominal under blocks (TV, 82%) is the shrinkage story in point 3, not a
resampling failure.

**2. Post-selection inference.** The transforms and the penalty are chosen by
search (:mod:`~mmm_framework.frequentist.search`). If every replicate conditions
on that one choice, the interval omits selection uncertainty — and §4a of the
spec measured that λ is not identified by the criterion *at all* (candidates
spanning ≈0.16–7.8 score within 10% of the winner). So the quantity being
conditioned on is not a well-estimated parameter with a little noise around it;
it is a near-arbitrary pick from a set the data cannot order.

``refit_search=True`` re-runs the search inside every replicate and is the
correct thing. It costs ``n_boot × search``. The default is ``False`` **because
an unaffordable honest default gets switched off and then the label goes with
it** — so the cheap path is labelled instead, at every surface that renders the
number: ``diagnostics["interval_semantics"]`` is ``"conditional_on_selection"``,
the same string rides in the ``InferenceData`` attrs, and #188 renders it.

**3. Ridge is biased.** A percentile bootstrap interval around a shrunk
estimator covers the *estimator's* sampling distribution, not the true
parameter. Bias correction (:func:`bc_interval`) and acceleration
(:func:`bca_interval`) fix the bootstrap distribution's median-bias and
skewness; neither removes shrinkage bias relative to the truth. Coverage for the
truth therefore falls below nominal exactly when the penalty is doing real work,
and the honest instrument for "how much work" is the effective degrees of freedom
reported by the ridge fit — carried through as ``diagnostics["effective_dof"]``.

Why the trace holds **percentile** draws
----------------------------------------
Downstream consumes draws, not intervals, and a BC/BCa adjustment is
per-statistic and per-level: it cannot be baked into a draw array without
changing what the draws mean. So the container carries the replicate
distribution as-is (``interval_kind="bootstrap_percentile"``), which is also
what ``diagnostics/coverage.py`` grades (central equal-tailed intervals, matching
``compute_hdi_bounds``) and what keeps the headline coverage number comparable to
the Bayesian path's. :func:`bc_interval` / :func:`bca_interval` are available for
a specific scalar, and are validated separately.

The container
-------------
Replicates are packaged ``(chain=1, draw=n_boot)``, following the
``arviz_compat.point_to_idata`` precedent that wraps a MAP point as
``(chain=1, draw=1)``. Every Deterministic downstream code reads —
``channel_contributions``, ``media_total``, ``controls_total``, ``y_obs_scaled``,
``beta_<ch>``, the component Deterministics — is **evaluated out of the model's
own graph** at each replicate's parameter vector, so they cannot drift from the
Bayesian definitions. The graph is compiled once, with the coefficients, the
transform point and the ROI scaling as symbolic inputs, so ``refit_search=True``
(a different transform point per replicate) costs no recompilation.

``run_smc_fit``, not ``run_approximate_fit``, is the structural precedent: an
estimator that is not approximate but is also not NUTS, whose estimator-specific
numbers ride in the returned extra-diagnostics dict. ``approximate`` stays
``False`` here — see ``technical-docs/frequentist-estimation.md`` §8 for why it
is the wrong flag to reuse.
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from .design import UnsupportedModelError, build_design_matrix
from .ridge import fit_ridge

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from ..data_loader import PanelDataset
    from .design import DesignMatrix

__all__ = [
    "bc_interval",
    "bca_interval",
    "bootstrap_fit",
    "estimate_block_length",
    "moving_block_indices",
    "residual_autocorrelation",
]

#: Variables whose only role is to reparameterize an identified composite. The
#: bootstrap solves for the composite (``geo_sigma * geo_offset``,
#: ``spline_scale * cumsum(spline_coef_raw)``), so writing a value for the parts
#: would publish an arbitrary factorization as if it were an estimate. The
#: composite is emitted instead — ``geo_effect`` / ``product_effect`` as extra
#: outputs, ``spline_coef`` because the graph already registers it.
_NUISANCE_PARTS = frozenset(
    {
        "geo_sigma",
        "geo_offset",
        "product_sigma",
        "product_offset",
        "spline_scale",
        "spline_coef_raw",
        "trend_m",
    }
)


# --------------------------------------------------------------------------- #
# block length
# --------------------------------------------------------------------------- #


def residual_autocorrelation(
    resid: "NDArray[np.floating]",
    time_idx: "NDArray[np.integer]",
    cell_idx: "NDArray[np.integer] | None" = None,
    n_cells: int = 1,
) -> float:
    """Pooled lag-1 autocorrelation of the residual series.

    Computed **within** each panel cell and pooled by summing the numerators and
    denominators, which is the panel AR(1) estimator. Pooling the *estimates*
    instead would weight a short geo the same as a long one, and stacking the
    cells into one flat series would manufacture a spurious lag-1 pair at every
    geo boundary.

    Args:
        resid: ``(n_obs,)`` residuals.
        time_idx: ``(n_obs,)`` period index per row.
        cell_idx: ``(n_obs,)`` panel-cell index per row. ``None`` = national.
        n_cells: Number of panel cells.

    Returns:
        ρ̂ clipped to ``[0, 0.99]``. Negative estimates return ``0.0``: a
        negatively autocorrelated residual needs no block (blocks exist to stop
        positive dependence being mistaken for independent information), and the
        MSE-optimal rule below is undefined for ρ < 0.
    """
    resid = np.asarray(resid, dtype=float)
    t = np.asarray(time_idx, dtype=int)
    cells = np.zeros_like(t) if cell_idx is None else np.asarray(cell_idx, dtype=int)
    num = 0.0
    den = 0.0
    for g in range(max(int(n_cells), 1)):
        rows = np.flatnonzero(cells == g)
        if rows.size < 3:
            continue
        series = resid[rows[np.argsort(t[rows])]]
        num += float(series[1:] @ series[:-1])
        den += float(series[:-1] @ series[:-1])
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 0.99))


def estimate_block_length(rho: float, n_periods: int) -> int:
    """MSE-optimal moving-block length for an AR(1) residual process.

    ``b = ceil(n^(1/3) · (2ρ̂/(1−ρ̂²))^(2/3))``, clipped to ``[1, n//4]``. The
    upper clip matters: an unclipped rule at ρ̂ → 1 asks for blocks longer than
    the series, which resamples one block and produces no variability at all.
    ``ρ̂ = 0`` gives ``b = 1``, i.e. the iid residual bootstrap — the comparison
    case, reachable on purpose with ``block_length=1``.
    """
    n = int(n_periods)
    if rho <= 0 or n < 4:
        return 1
    b = math.ceil((n ** (1.0 / 3.0)) * ((2.0 * rho / (1.0 - rho**2)) ** (2.0 / 3.0)))
    return int(np.clip(b, 1, max(1, n // 4)))


def moving_block_indices(
    n_periods: int, block_length: int, rng: np.random.Generator
) -> "NDArray[np.integer]":
    """A length-``n_periods`` period index drawn from **overlapping** blocks.

    Overlapping (moving) rather than disjoint blocks: every position is an
    equally likely block start, so no period is systematically over- or
    under-represented by where the partition happened to fall.
    """
    n = int(n_periods)
    b = int(np.clip(block_length, 1, max(n, 1)))
    if b <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(math.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(b)[None, :]).ravel()
    return idx[:n]


def _rows_by_cell(
    time_idx: "NDArray[np.integer]",
    cell_idx: "NDArray[np.integer]",
    n_cells: int,
) -> list["NDArray[np.integer]"]:
    """Row indices per panel cell, ordered in time. National = one entry."""
    out = []
    for g in range(max(int(n_cells), 1)):
        rows = np.flatnonzero(cell_idx == g)
        out.append(rows[np.argsort(time_idx[rows])])
    return [r for r in out if r.size]


def _resample_residuals(
    resid: "NDArray[np.floating]",
    rows_by_cell: list["NDArray[np.integer]"],
    block: int,
    rng: np.random.Generator,
) -> "NDArray[np.floating]":
    """One moving-block residual resample, **synchronized across panel cells**.

    On a rectangular panel every cell reuses the same resampled position
    sequence, so a resampled week carries every geography's residual together.
    Resampling geographies independently would destroy the contemporaneous
    cross-sectional correlation that is exactly what makes a panel more
    informative than one national series.

    A ragged panel (cells with different period coverage) has no shared position
    sequence, so each cell gets its own block draw — the marginal variance is
    right, the cross-sectional coupling is not preserved, and
    :func:`bootstrap_fit` says so.
    """
    out = np.empty_like(resid)
    lengths = {len(rows) for rows in rows_by_cell}
    if len(lengths) == 1:
        pos = moving_block_indices(lengths.pop(), block, rng)
        for rows in rows_by_cell:
            out[rows] = resid[rows[pos]]
    else:
        for rows in rows_by_cell:
            out[rows] = resid[rows[moving_block_indices(len(rows), block, rng)]]
    return out


# --------------------------------------------------------------------------- #
# bias-corrected intervals (opt-in, per statistic)
# --------------------------------------------------------------------------- #


def _norm_ppf(p: "NDArray[np.floating] | float") -> "NDArray[np.floating]":
    from scipy import stats

    return np.asarray(stats.norm.ppf(p), dtype=float)


def _norm_cdf(x: "NDArray[np.floating] | float") -> "NDArray[np.floating]":
    from scipy import stats

    return np.asarray(stats.norm.cdf(x), dtype=float)


def bc_interval(
    draws: "NDArray[np.floating]", point: float, level: float = 0.9
) -> tuple[float, float]:
    """Bias-corrected (BC) percentile interval for one scalar statistic.

    Corrects the **median bias of the bootstrap distribution** — the fraction of
    replicates falling below the point estimate becomes ``z0`` and shifts the
    percentile levels. It does **not** remove ridge shrinkage bias relative to
    the true parameter; nothing does. See the module docstring.

    Degenerate cases (every replicate on one side of the point estimate, which
    happens for a coefficient pinned at a non-negativity boundary) fall back to
    the plain percentile interval rather than returning an infinite endpoint.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    lo_p, hi_p = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    if d.size < 2:
        return (float("nan"), float("nan"))
    frac = float(np.mean(d < point))
    if not 0.0 < frac < 1.0:
        return (float(np.quantile(d, lo_p)), float(np.quantile(d, hi_p)))
    z0 = float(_norm_ppf(frac))
    lo = float(_norm_cdf(2 * z0 + _norm_ppf(lo_p)))
    hi = float(_norm_cdf(2 * z0 + _norm_ppf(hi_p)))
    return (float(np.quantile(d, lo)), float(np.quantile(d, hi)))


def bca_interval(
    draws: "NDArray[np.floating]",
    point: float,
    jackknife: "NDArray[np.floating]",
    level: float = 0.9,
) -> tuple[float, float]:
    """BCa interval — BC plus an acceleration term for a skewed statistic.

    ``jackknife`` is the leave-one-out statistic array whose third/second moment
    ratio estimates the acceleration ``a``. Falls back to :func:`bc_interval`
    when the jackknife spread is degenerate.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    j = np.asarray(jackknife, dtype=float)
    j = j[np.isfinite(j)]
    if d.size < 2 or j.size < 3:
        return bc_interval(draws, point, level)
    dev = j.mean() - j
    denom = 6.0 * float(np.sum(dev**2)) ** 1.5
    if denom <= 0:
        return bc_interval(draws, point, level)
    a = float(np.sum(dev**3)) / denom
    frac = float(np.mean(d < point))
    if not 0.0 < frac < 1.0:
        return bc_interval(draws, point, level)
    z0 = float(_norm_ppf(frac))
    out = []
    for p in ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0):
        z = z0 + _norm_ppf(p)
        adj = float(_norm_cdf(z0 + z / (1.0 - a * z)))
        out.append(float(np.quantile(d, np.clip(adj, 1e-6, 1 - 1e-6))))
    return (out[0], out[1])


# --------------------------------------------------------------------------- #
# replicate -> trace
# --------------------------------------------------------------------------- #


def _like(var: Any, value: float) -> Any:
    """A constant matching ``var``'s dtype.

    PyTensor's ``as_tensor`` uses ``floatX`` (float32 on this project's default
    config) while the model's RVs are float64, and ``graph_replace`` refuses a
    dtype mismatch outright.
    """
    import pytensor.tensor as pt

    return pt.constant(np.asarray(value, dtype=var.dtype))


def _column_groups(design: "DesignMatrix") -> dict[str, list[int]]:
    """Design column indices grouped by the graph parameter they feed, in order."""
    groups: dict[str, list[tuple[int, int]]] = {}
    for col_i, (param, pos) in design.param_map.items():
        groups.setdefault(param, []).append((pos, col_i))
    return {p: [c for _, c in sorted(v)] for p, v in groups.items()}


def _transform_slots(
    graph: Any, channels: "Sequence[str]"
) -> list[tuple[str, str, str]]:
    """``(rv_name, channel, source_key)`` for every fixed-transform free RV.

    ``source_key`` indexes ``alpha[channel]`` (``alpha`` / ``theta`` / ``shape`` /
    ``scale``) or ``lam[channel]`` (which already carries the ``sat_`` prefix the
    graph uses). Channels are matched longest-first so ``"TV"`` cannot claim
    ``adstock_alpha_CTV``.
    """
    by_len = sorted(channels, key=len, reverse=True)
    slots: list[tuple[str, str, str]] = []
    for rv in graph.free_RVs:
        name = rv.name
        if not (name.startswith("adstock_") or name.startswith("sat_")):
            continue
        for ch in by_len:
            if name.endswith(f"_{ch}"):
                stem = name[: -(len(ch) + 1)]
                key = stem[len("adstock_") :] if stem.startswith("adstock_") else stem
                slots.append((name, ch, key))
                break
    return slots


class _ReplicateEvaluator:
    """Evaluate the model graph at a replicate's parameter vector.

    Compiled **once**, with the coefficient vector, the residual scale, the
    transform point and the ROI scaling as symbolic inputs — so a
    ``refit_search=True`` run, whose transform point differs every replicate,
    costs no recompilation.

    The replacement set is the inverse of the mapping
    ``tests/frequentist/test_design_equivalence.py`` verifies in the forward
    direction, which is why the two cannot drift: geo/product enter the graph as
    ``sigma * offset`` and are replaced by ``1 * theta`` (an equivalent
    factorization of the identified composite), the spline's registered
    ``spline_coef`` Deterministic is replaced outright, and ``trend_m`` — exactly
    collinear with the intercept — is set to zero because the design folds it
    into the intercept.
    """

    def __init__(self, mmm: Any, design: "DesignMatrix") -> None:
        import pytensor
        import pytensor.tensor as pt
        from pytensor.graph.replace import graph_replace

        graph = mmm.model
        self._graph = graph
        self.channels = list(mmm.channel_names)
        self.transform_slots = _transform_slots(graph, self.channels)

        theta = pt.dvector("theta")
        sigma = pt.dscalar("sigma")
        trans = pt.dvector("transform")
        roi = pt.dvector("roi_scale")

        named = graph.named_vars
        groups = _column_groups(design)
        repl: dict[Any, Any] = {}
        extra: dict[str, Any] = {}

        for param, cols in groups.items():
            idx = np.asarray(cols, dtype=int)
            if param in ("geo_effect", "product_effect"):
                level = param.split("_")[0]
                repl[named[f"{level}_sigma"]] = _like(named[f"{level}_sigma"], 1.0)
                repl[named[f"{level}_offset"]] = theta[idx]
                extra[param] = theta[idx]
                continue
            if param.startswith("beta_") and param != "beta_controls":
                ch = param[len("beta_") :]
                c = self.channels.index(ch)
                repl[named[param]] = roi[c] * theta[idx[0]]
                roi_name = f"roi_{ch}"
                if roi_name in named:
                    extra[roi_name] = theta[idx[0]]
                continue
            var = named.get(param)
            if var is None:  # pragma: no cover - design/graph drift guard
                raise UnsupportedModelError(
                    f"Design parameter {param!r}",
                    "it has no matching variable in the model graph, so a "
                    "bootstrap replicate cannot be written back into it",
                )
            repl[var] = theta[idx] if var.ndim else theta[idx[0]]

        # `trend_m` is folded into the (unpenalized) intercept by the design.
        if "trend_m" in named:
            repl[named["trend_m"]] = _like(named["trend_m"], 0.0)
        if "sigma" in named:
            repl[named["sigma"]] = sigma
        for pos, (rv_name, _ch, _key) in enumerate(self.transform_slots):
            repl[named[rv_name]] = trans[pos]

        unresolved = sorted(
            rv.name
            for rv in graph.free_RVs
            if rv not in repl and rv.name not in _NUISANCE_PARTS
        )
        # Nuisance parts drop out of the graph once their composite is replaced,
        # so they only *look* unresolved; anything else is a real gap.
        outputs_names: list[str] = []
        outputs: list[Any] = []
        for det in graph.deterministics:
            outputs_names.append(det.name)
            outputs.append(det)
        replaced_outputs = graph_replace(outputs, repl, strict=False)
        if unresolved:
            still = {
                v.name for v in _ancestors(replaced_outputs) if v in set(graph.free_RVs)
            }
            missing = sorted(n for n in unresolved if n in still)
            if missing:
                raise UnsupportedModelError(
                    "Free parameters outside the design",
                    "the bootstrap has no value for "
                    + ", ".join(missing)
                    + "; this configuration is not a fixed-transform linear model",
                )

        for name, expr in extra.items():
            outputs_names.append(name)
            replaced_outputs.append(expr)
        for var, expr in repl.items():
            if var.name in _NUISANCE_PARTS or var.name in outputs_names:
                continue
            outputs_names.append(var.name)
            replaced_outputs.append(pt.as_tensor(expr))

        self.names = outputs_names
        self._fn = pytensor.function(
            [theta, sigma, trans, roi],
            replaced_outputs,
            on_unused_input="ignore",
        )

    def transform_vector(
        self, alpha: "Mapping[str, Mapping[str, float]]", lam: "Mapping[str, Any]"
    ) -> "NDArray[np.floating]":
        """Pack a ``(alpha, lam)`` point into the compiled function's input order."""
        out = np.empty(len(self.transform_slots), dtype=float)
        for i, (rv_name, ch, key) in enumerate(self.transform_slots):
            source = (
                alpha.get(ch, {}) if rv_name.startswith("adstock_") else lam.get(ch, {})
            )
            if key not in source:
                raise KeyError(
                    f"transform point is missing {key!r} for channel {ch!r} "
                    f"(needed by {rv_name!r})"
                )
            out[i] = float(source[key])
        return out

    def __call__(
        self,
        theta: "NDArray[np.floating]",
        sigma: float,
        transform: "NDArray[np.floating]",
        roi_scale: "NDArray[np.floating]",
    ) -> dict[str, "NDArray[np.floating]"]:
        values = self._fn(
            np.asarray(theta, dtype=float),
            float(sigma),
            np.asarray(transform, dtype=float),
            np.asarray(roi_scale, dtype=float),
        )
        return dict(zip(self.names, values, strict=True))


def _ancestors(outputs):
    try:  # pytensor moved `ancestors` out of graph.basic
        from pytensor.graph.traversal import ancestors
    except ImportError:  # pragma: no cover - older pytensor
        from pytensor.graph.basic import ancestors
    return list(ancestors(outputs))


def _to_idata(
    graph: Any, draws: dict[str, list["NDArray[np.floating]"]], attrs: dict[str, Any]
):
    """Stack per-replicate values into a ``(chain=1, draw=n_boot)`` container.

    Dims and coords are taken from the **model**, so a bootstrap trace indexes
    like a NUTS trace (``channel_contributions`` really has ``channel``
    coordinates) rather than like a bag of anonymous axes.
    """
    import xarray as xr

    from ..utils import arviz_compat

    n_draws = len(next(iter(draws.values()))) if draws else 0
    var_dims = getattr(graph, "named_vars_to_dims", {}) or {}
    model_coords = {
        k: list(v)
        for k, v in (getattr(graph, "coords", {}) or {}).items()
        if v is not None
    }

    data_vars: dict[str, Any] = {}
    used_coords: dict[str, Any] = {"chain": [0], "draw": list(range(n_draws))}
    for name, series in draws.items():
        arr = np.asarray(series, dtype=float)[np.newaxis, ...]  # (1, draw, *shape)
        # PyMC accepts `dims="control"` as a bare STRING as well as a 1-tuple,
        # and iterating a string yields characters — so the length check below
        # would silently fall through to anonymous axes. A variable with NO
        # declared dims legitimately gets anonymous ones; that matches what a
        # NUTS trace of the same model carries.
        declared = var_dims.get(name) or ()
        if isinstance(declared, str):
            declared = (declared,)
        dims = [d for d in declared if d]
        if len(dims) == arr.ndim - 2 and all(
            d in model_coords and len(model_coords[d]) == arr.shape[2 + i]
            for i, d in enumerate(dims)
        ):
            axes = list(dims)
            for d in dims:
                used_coords[d] = model_coords[d]
        else:
            axes = [f"{name}_dim_{i}" for i in range(arr.ndim - 2)]
        data_vars[name] = (["chain", "draw", *axes], arr)

    ds = xr.Dataset(data_vars, coords=used_coords)
    # netCDF has no boolean attribute type, and `bool` is an `int` subclass —
    # so an unguarded isinstance check lets `approximate: False` through and the
    # first `trace.to_netcdf()` raises. Booleans are stringified rather than
    # dropped: they are the provenance flags a reader most wants to see.
    ds.attrs.update(
        {
            k: (str(v) if isinstance(v, bool) else v)
            for k, v in attrs.items()
            if isinstance(v, (str, int, float))
        }
    )
    return arviz_compat.dataset_to_idata(ds)


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #


def _panel_with_y(panel: "PanelDataset", y_raw: "NDArray[np.floating]"):
    import pandas as pd

    return dataclasses.replace(
        panel,
        y=pd.Series(
            np.asarray(y_raw, dtype=float), index=panel.y.index, name=panel.y.name
        ),
    )


class _Solve:
    """One replicate's solve — ridge, or the constrained QP when asked.

    Both estimators take ``(design, y, penalty)`` and return a coefficient
    vector, so the bootstrap loop does not branch. ``ConstrainedFit`` carries no
    ``residual_sd`` or ``effective_dof``, so those are derived here using the
    **unconstrained** ridge dof — an upper bound, since an active constraint can
    only remove freedom. It is used to inflate the residuals and to describe how
    hard the penalty is working, and both readings stay conservative under an
    upper bound.
    """

    def __init__(self, penalty: float, nonneg, constraints) -> None:
        self.penalty = float(penalty)
        self.nonneg = nonneg
        self.constraints = list(constraints or [])
        self.estimator = "constrained" if self.constraints else "ridge"

    def __call__(self, design: "DesignMatrix", y=None, penalty: float | None = None):
        pen = self.penalty if penalty is None else float(penalty)
        if not self.constraints:
            fit = fit_ridge(design, y=y, penalty=pen, nonneg=self.nonneg)
            return fit.theta, fit.residual_sd, fit.at_boundary, fit.effective_dof

        from .ridge import _effective_dof
        from .constrained import fit_constrained

        fit = fit_constrained(
            design, y=y, penalty=pen, constraints=self.constraints
        )
        y_vec = np.asarray(design.y if y is None else y, dtype=float)
        resid = y_vec - design.X @ fit.theta
        edof = _effective_dof(
            np.asarray(design.X, dtype=float), pen, design.penalize.astype(float)
        )
        n = design.n_obs
        sd = float(np.sqrt(float(resid @ resid) / max(n - edof, 1.0)))
        return fit.theta, sd, fit.at_boundary, edof


def bootstrap_fit(
    panel: "PanelDataset",
    *,
    model_config: Any,
    trend_config: Any,
    alpha: "Mapping[str, Mapping[str, float]] | None" = None,
    lam: "Mapping[str, Mapping[str, float]] | None" = None,
    penalty: float | None = None,
    n_boot: int = 500,
    block_length: int | None = None,
    refit_search: bool = False,
    nonneg: "bool | Sequence[str]" = False,
    constraints: "Sequence[Any] | None" = None,
    search_kwargs: "Mapping[str, Any] | None" = None,
    seed: int | None = None,
    progress: "Callable[[int, int], None] | None" = None,
):
    """Moving-block residual bootstrap for the frequentist path.

    Args:
        panel: The panel to fit.
        model_config: The :class:`ModelConfig` the fit uses. ``ridge_alpha`` is
            the penalty fallback when ``penalty`` and the search are both absent.
        trend_config: The :class:`TrendConfig` the fit uses.
        alpha: Per-channel adstock parameters to hold fixed. ``None`` runs
            :func:`~mmm_framework.frequentist.search.search_transforms`.
        lam: Per-channel saturation parameters to hold fixed. Must accompany
            ``alpha``.
        penalty: Ridge strength. ``None`` takes the search's selection, or
            ``model_config.ridge_alpha`` when the transforms were given
            explicitly.
        n_boot: Replicate count. Each replicate is one linear solve, so this is
            cheap unless ``refit_search`` is on.
        block_length: Moving-block length in **periods**. ``None`` estimates it
            from the residual autocorrelation (:func:`estimate_block_length`).
            Pass ``1`` for the iid residual bootstrap — the under-covering
            comparison case, useful for demonstrating the difference and wrong
            for reporting.
        refit_search: Re-run the transform search inside every replicate, so the
            interval includes selection uncertainty. Correct and expensive; see
            the module docstring for why it is not the default. Requires trend
            ``none`` or ``linear`` (the search's own restriction).
        nonneg: Passed to :func:`~mmm_framework.frequentist.ridge.fit_ridge`.
            ``True`` constrains the media block to be non-negative. Ignored when
            ``constraints`` is supplied (express it as a constraint instead).
        constraints: Linear constraints from
            :mod:`~mmm_framework.frequentist.constrained`, or a **factory**
            ``(design) -> list[Constraint]`` for the common case where the
            estimator is chosen before a transform point exists (a Constraint's
            row is indexed by design column). Non-empty switches
            every replicate to the convex program and stamps
            ``estimator="constrained"``; this is what ``frequentist_cvxpy``
            reaches, and it needs the optional ``[frequentist]`` extra. A
            coefficient pinned by an active constraint has **no meaningful
            two-sided interval** — those columns are listed in
            ``diagnostics["at_boundary"]``.
        search_kwargs: Extra keyword arguments for ``search_transforms``.
        seed: Seed for the resampling (and for the search, when it runs).
        progress: Optional ``(done, total)`` callback, called per replicate.

    Returns:
        ``(idata, diagnostics)``. ``idata`` is a ``(chain=1, draw=n_boot)``
        container carrying the model's Deterministics, so ``predict`` /
        reporting / the estimand engine work unchanged. ``diagnostics`` carries
        the provenance contract of ``technical-docs/frequentist-estimation.md``
        §8 — ``inference_family``, ``interval_kind``, ``interval_semantics`` and
        the numbers behind the interval — and is JSON-safe.

    Raises:
        ValueError: If ``alpha`` and ``lam`` disagree about being supplied, or
            ``n_boot < 2``.
        UnsupportedModelError: If the configuration is not a fixed-transform
            linear model.
    """
    from ..model.base import BayesianMMM

    t0 = time.perf_counter()
    if (alpha is None) != (lam is None):
        raise ValueError(
            "alpha and lam must be supplied together (or both omitted to search)"
        )
    if int(n_boot) < 2:
        raise ValueError(f"n_boot must be at least 2, got {n_boot}")
    n_boot = int(n_boot)

    trend_type = str(getattr(trend_config.type, "value", trend_config.type))
    search_kwargs = dict(search_kwargs or {})
    rng = np.random.default_rng(seed)

    # -- 0. refuse early -------------------------------------------------------
    # Before the search, so the message names the MODEL feature that is not
    # linear rather than the search step that happened to hit it first.
    from ..model.base import BayesianMMM as _BMMM

    from .design import _reject_unsupported

    _reject_unsupported(_BMMM(panel, model_config, trend_config))

    # -- 1. the transform point ------------------------------------------------
    search_result = None
    if alpha is None:
        from .search import search_transforms

        search_kwargs.setdefault("seed", int(seed or 0))
        search_result = search_transforms(
            panel,
            model_config=model_config,
            trend_config=trend_config,
            **search_kwargs,
        )
        alpha = search_result.best.alpha
        lam = search_result.best.lam
        if penalty is None:
            penalty = search_result.best.penalty
        criterion = search_result.criterion
    else:
        criterion = "fixed"
    if penalty is None:
        penalty = float(getattr(model_config, "ridge_alpha", 0.0))

    if refit_search and trend_type not in ("none", "linear"):
        raise UnsupportedModelError(
            f"refit_search with a {trend_type} trend",
            "the search scores out of time and those bases do not extrapolate; "
            "use trend 'none' or 'linear', or accept the conditional-on-"
            "selection interval",
        )

    # -- 2. the point estimate -------------------------------------------------
    mmm = BayesianMMM(panel, model_config, trend_config)
    design = build_design_matrix(
        panel,
        dict(alpha),
        dict(lam),
        model_config=model_config,
        trend_config=trend_config,
    )
    if callable(constraints):
        # `frequentist_cvxpy` selects the estimator before a transform point
        # exists, and a Constraint's `a` vector is indexed by design column — so
        # the constraint set is supplied as a factory and resolved here, once the
        # search has produced a design. Column ORDER is a function of the config,
        # not of the transform point, so a `refit_search` rebuild reuses these
        # rows safely.
        constraints = constraints(design)
    solve = _Solve(float(penalty), nonneg, constraints)
    theta_hat, base_sd, base_boundary, base_edof = solve(design)

    fitted = design.X @ theta_hat
    resid = design.y - fitted
    resid = resid - resid.mean()
    # Residuals from a penalized fit are shrunk toward the fitted surface, so
    # resampling them raw understates the noise. The OLS `sqrt(n/(n-p))`
    # correction generalizes with the EFFECTIVE degrees of freedom, which is
    # exactly what makes `effective_dof` load-bearing rather than decorative.
    n_obs = int(design.n_obs)
    inflate = math.sqrt(n_obs / max(n_obs - base_edof, 1.0))
    resid = resid * inflate

    # -- 3. the block length ---------------------------------------------------
    time_idx = np.asarray(mmm.time_idx, dtype=int)
    cell_idx = np.asarray(
        getattr(mmm, "cell_idx", np.zeros(n_obs, dtype=int)), dtype=int
    )
    n_cells = int(getattr(mmm, "n_cells", 1) or 1)
    n_periods = int(mmm.n_periods)
    rho = residual_autocorrelation(resid, time_idx, cell_idx, n_cells)
    if block_length is None:
        block = estimate_block_length(rho, n_periods)
        block_source = "estimated"
    else:
        block = int(np.clip(int(block_length), 1, max(n_periods, 1)))
        block_source = "explicit"

    rows_by_cell = _rows_by_cell(time_idx, cell_idx, n_cells)
    ragged = len({len(r) for r in rows_by_cell}) > 1
    if ragged:
        warnings.warn(
            "Panel cells have different period coverage, so residual blocks are "
            "drawn per cell: the marginal variance is right but the "
            "contemporaneous cross-sectional correlation is not preserved.",
            stacklevel=2,
        )

    # -- 4. replicates ---------------------------------------------------------
    evaluator = _ReplicateEvaluator(mmm, design)
    base_transform = evaluator.transform_vector(alpha, lam)
    base_roi = np.array([design.roi_scale.get(c, 1.0) for c in evaluator.channels])

    draws: dict[str, list[np.ndarray]] = {}
    n_failed = 0
    y_mean, y_std = float(design.scaling["y_mean"]), float(design.scaling["y_std"])
    for b in range(n_boot):
        y_star = fitted + _resample_residuals(resid, rows_by_cell, block, rng)

        rep_design, rep_penalty = design, float(penalty)
        rep_transform, rep_roi = base_transform, base_roi
        try:
            if refit_search:
                from .search import search_transforms

                rep_search = search_transforms(
                    _panel_with_y(panel, y_star * y_std + y_mean),
                    model_config=model_config,
                    trend_config=trend_config,
                    **{**search_kwargs, "seed": int(rng.integers(0, 2**31 - 1))},
                )
                rep_alpha, rep_lam = rep_search.best.alpha, rep_search.best.lam
                rep_penalty = float(rep_search.best.penalty)
                # Rebuilt on the ORIGINAL panel so every replicate shares one
                # standardization: selection uncertainty propagates, the scale
                # the penalty and the coefficients live on does not move.
                rep_design = build_design_matrix(
                    panel,
                    dict(rep_alpha),
                    dict(rep_lam),
                    model_config=model_config,
                    trend_config=trend_config,
                )
                rep_transform = evaluator.transform_vector(rep_alpha, rep_lam)
                rep_roi = np.array(
                    [rep_design.roi_scale.get(c, 1.0) for c in evaluator.channels]
                )
            rep_theta, rep_sd, _, _ = solve(rep_design, y_star, rep_penalty)
            values = evaluator(rep_theta, rep_sd, rep_transform, rep_roi)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one bad replicate must not kill the fit
            n_failed += 1
            if n_failed == 1:
                warnings.warn(
                    f"bootstrap replicate {b} failed and was dropped ({exc}); "
                    "the interval is built from the survivors.",
                    stacklevel=2,
                )
            if progress:
                progress(b + 1, n_boot)
            continue

        for name, value in values.items():
            draws.setdefault(name, []).append(np.asarray(value, dtype=float))
        if progress:
            progress(b + 1, n_boot)

    if not draws:
        raise RuntimeError(
            f"every one of the {n_boot} bootstrap replicates failed; no interval "
            "could be formed"
        )
    n_effective = len(next(iter(draws.values())))

    semantics = "selection_resampled" if refit_search else "conditional_on_selection"
    diagnostics: dict[str, Any] = {
        # -- provenance contract (technical-docs/frequentist-estimation.md §8) --
        "inference_family": "frequentist",
        "estimator": solve.estimator,
        "interval_kind": "bootstrap_percentile",
        "interval_semantics": semantics,
        "selection_criterion": criterion,
        "approximate": False,
        "fit_method": None,
        # -- the numbers behind the interval --
        "n_boot": n_boot,
        "n_boot_effective": int(n_effective),
        "n_boot_failed": int(n_failed),
        "block_length": int(block),
        "block_length_source": block_source,
        "block_synchronized": not ragged,
        "residual_rho": float(rho),
        "residual_inflation": float(inflate),
        "penalty": float(penalty),
        "effective_dof": float(base_edof),
        "n_params": int(design.n_params),
        "n_obs": n_obs,
        "residual_sd": float(base_sd),
        "nonneg": bool(nonneg),
        "constraints": [str(getattr(c, "label", c)) for c in (constraints or [])],
        "at_boundary": [
            c for c, on in zip(design.columns, base_boundary, strict=False) if on
        ],
        "transform_alpha": {k: dict(v) for k, v in dict(alpha).items()},
        "transform_lam": {k: dict(v) for k, v in dict(lam).items()},
        "point_estimate": dict(
            zip(design.columns, (float(v) for v in theta_hat), strict=False)
        ),
        "elapsed_s": float(time.perf_counter() - t0),
        "caveats": _caveats(
            semantics, base_edof, design.n_params, criterion, search_result
        ),
    }
    if search_result is not None:
        near = search_result.spread(0.10)
        diagnostics["search"] = {
            "criterion": search_result.criterion,
            "budget": len(search_result.candidates),
            "n_within_10pct": len(near),
            "best_score": float(search_result.best.score),
            "caveat": search_result.caveat,
        }

    return _to_idata(mmm.model, draws, diagnostics), diagnostics


def _caveats(
    semantics: str,
    effective_dof: float,
    n_params: int,
    criterion: str,
    search_result: Any,
) -> list[str]:
    """The statements that must ride with any rendered interval from this path."""
    out = [
        "These are bootstrap CONFIDENCE intervals from a frequentist point "
        "estimate, not credible intervals: they describe the sampling "
        "variability of the estimator, not a probability distribution over the "
        "parameter. 'There is a 90% probability the ROI is in this range' is "
        "false for them.",
    ]
    shrunk = effective_dof < 0.95 * n_params
    out.append(
        "Ridge is biased by construction, so a percentile interval covers the "
        "estimator's sampling distribution rather than the true parameter. The "
        f"penalty is currently using {effective_dof:.1f} of "
        f"{n_params} effective degrees of freedom"
        + (
            " — it is doing real work here, so read coverage for the truth as "
            "below nominal."
            if shrunk
            else " — close to unpenalized, so shrinkage bias is small."
        )
    )
    if semantics == "conditional_on_selection":
        out.append(
            "The adstock/saturation transforms and the penalty were selected "
            f"once ({criterion}) and every replicate conditions on that choice, "
            "so this interval OMITS selection uncertainty and is too narrow. "
            "Re-run with refit_search=True for an interval that includes it."
        )
        if search_result is not None:
            near = len(search_result.spread(0.10))
            if near > 1:
                out.append(
                    f"{near} candidate transform points scored within 10% of the "
                    "winner — the data cannot order them, so the conditioned-on "
                    "choice is close to arbitrary."
                )
    else:
        out.append(
            "The transform search was re-run inside every replicate, so this "
            "interval includes selection uncertainty."
        )
    if criterion != "fixed":
        out.append(
            "Transforms were selected by out-of-sample predictive error, which "
            "is not a causal criterion: a specification that predicts better "
            "can attribute worse."
        )
    return out
