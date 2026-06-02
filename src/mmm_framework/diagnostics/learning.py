"""Prior-vs-posterior *learning* diagnostics.

A posterior credible interval only tells you what you believe *after* seeing the
data -- it does not tell you how much of that belief came from the **data** versus
the **prior**. When a prior is informative or sign-constrained, a posterior can
look "conclusive" while encoding almost nothing the prior did not already assert.

The canonical example in this framework is the cannibalization cross-effect of
:class:`~mmm_framework.mmm_extensions.models.MultivariateMMM`, whose prior is
``psi = -HalfNormal(sigma)`` -- *structurally* non-positive. Reporting that the
posterior of ``psi`` is "entirely below zero" is then nearly vacuous: so is the
prior. The honest question is whether the data **moved** or **narrowed** that
parameter relative to its prior.

This module quantifies that with three complementary, sample-based diagnostics:

* **contraction** ``c = 1 - Var_post / Var_prior`` (Betancourt; Schad, Betancourt &
  Vasishth 2021). ``c -> 1`` the posterior is far narrower than the prior (the data
  pinned it); ``c ~ 0`` the posterior is as wide as the prior (data uninformative);
  ``c < 0`` the posterior is *wider* than the prior -- a red flag for prior-data
  conflict or poor sampling (we deliberately do **not** clip it). Contraction is a
  pure variance ratio, so it is robust even for bounded/half-normal priors.
* **overlap** -- the prior-posterior overlap coefficient ``OVL = sum_i min(p_i, q_i)``
  over shared histogram bins (probability-mass form, in ``[0, 1]``). ``OVL ~ 1`` the
  posterior is indistinguishable from the prior (nothing learned); ``OVL ~ 0`` they
  barely overlap (strong learning, whether by narrowing *or* by shifting). Histogram
  binning -- not a Gaussian KDE -- is used on purpose so a hard prior edge (e.g. the
  ``HalfNormal`` boundary at 0) is not smeared.
* **shift_z** ``= (mean_post - mean_prior) / sd_prior`` -- how far, in prior standard
  deviations, the posterior mean moved. Catches pure *location* learning that
  contraction alone misses (a posterior can shift without narrowing).

Each parameter gets a heuristic ``verdict`` -- ``"strong"`` / ``"moderate"`` /
``"weak"`` / ``"prior-dominated"`` -- from ``contraction`` and ``overlap`` (the
thresholds are tunable; treat them as conventions, not law). A ``"prior-dominated"``
verdict is the diagnostic the user is after: the posterior is essentially the prior.

The core :func:`parameter_learning` works on raw samples (arviz ``InferenceData``,
xarray ``Dataset``, or plain dicts of arrays), so it is unit-testable without a fit.
The model classes expose ``compute_parameter_learning(...)`` which draws prior samples
and calls it for you.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = [
    "parameter_learning",
    "plot_parameter_learning",
    "plot_prior_posterior_overlay",
]

# Default histogram resolution for the overlap coefficient.
_DEFAULT_BINS = 60
# Verdict colors (matplotlib-friendly), surfaced so plots read consistently.
_VERDICT_COLORS = {
    "strong": "#3f7d5e",       # green  -- data clearly informed the parameter
    "moderate": "#3b6ea5",     # blue
    "weak": "#d98a2b",         # amber
    "prior-dominated": "#a63a50",  # red -- posterior ~ prior
    "undetermined": "#8a8079", # grey   -- degenerate prior (sd 0)
}


# =============================================================================
# Sample extraction (idata / dataset / dict -> {flat_param_name: 1d samples})
# =============================================================================


def _extract(obj: Any, group: str) -> dict[str, np.ndarray]:
    """Return ``{flat_parameter_name: 1d ndarray}`` from a samples container.

    Accepts an arviz ``InferenceData`` (the ``group`` -- ``"prior"`` or
    ``"posterior"`` -- is selected), an xarray ``Dataset``, or a plain ``dict`` of
    arrays. Vector/matrix parameters are flattened element-wise with a consistent
    ``name[i,j]`` convention so the prior and posterior align exactly.
    """
    # arviz InferenceData FIRST -- it is itself Mapping-like (over its groups), so it
    # must be unwrapped to the requested group before the plain-dict branch below.
    try:
        import arviz as az

        if isinstance(obj, az.InferenceData):
            if group not in obj.groups():
                raise KeyError(
                    f"InferenceData has no '{group}' group (groups: {obj.groups()})"
                )
            obj = obj[group]
    except ImportError:  # pragma: no cover - arviz is a hard dep in practice
        pass

    if hasattr(obj, "data_vars"):  # xarray Dataset (also Mapping-like -> check first)
        out: dict[str, np.ndarray] = {}
        for name in obj.data_vars:
            da = obj[name]
            dims = list(da.dims)
            sample_dims = [d for d in ("chain", "draw") if d in dims]
            if not sample_dims:
                continue  # not a sampled variable
            param_dims = [d for d in dims if d not in sample_dims]
            da = da.transpose(*sample_dims, *param_dims)
            values = np.asarray(da.values, dtype=float)
            n_samples = int(np.prod([da.sizes[d] for d in sample_dims]))
            param_shape = tuple(da.sizes[d] for d in param_dims)
            flat = values.reshape((n_samples, *param_shape))
            out.update(_flatten_array(str(name), flat, n_samples_axis=0, samples_first=True))
        return out

    if isinstance(obj, Mapping):  # plain dict of arrays (sample axis first)
        out = {}
        for name, arr in obj.items():
            out.update(_flatten_array(str(name), np.asarray(arr, dtype=float), 1))
        return out

    raise TypeError(
        "Expected an arviz InferenceData, xarray Dataset, or dict of arrays; "
        f"got {type(obj)!r}."
    )


def _flatten_array(
    name: str,
    arr: np.ndarray,
    n_samples_axis: int = 1,
    *,
    samples_first: bool = True,
) -> dict[str, np.ndarray]:
    """Flatten one (possibly vector/matrix) parameter into per-element sample arrays.

    ``arr`` has the sample axis at position 0 (``samples_first``); any remaining axes
    index the parameter's elements, named ``name[i,j,...]`` (or just ``name`` if
    scalar).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim <= 1:
        return {name: arr.reshape(-1)}
    if not samples_first:  # pragma: no cover - kept for symmetry
        arr = np.moveaxis(arr, n_samples_axis, 0)
    n = arr.shape[0]
    param_shape = arr.shape[1:]
    out: dict[str, np.ndarray] = {}
    for idx in np.ndindex(param_shape):
        key = f"{name}[{','.join(map(str, idx))}]"
        out[key] = arr[(slice(None), *idx)].reshape(-1)
    return out


# =============================================================================
# Metrics
# =============================================================================


def _overlap_coefficient(prior: np.ndarray, post: np.ndarray, bins: int) -> float:
    """Prior-posterior overlap coefficient via shared histogram bins, in ``[0, 1]``.

    Uses probability-mass histograms (not a Gaussian KDE) so a hard prior boundary
    -- e.g. the ``HalfNormal`` edge at 0 behind the cannibalization prior -- is not
    artificially smeared across the support.
    """
    prior = prior[np.isfinite(prior)]
    post = post[np.isfinite(post)]
    if prior.size == 0 or post.size == 0:
        return float("nan")
    lo = float(min(prior.min(), post.min()))
    hi = float(max(prior.max(), post.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(prior, bins=edges)
    q, _ = np.histogram(post, bins=edges)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.minimum(p, q).sum())


def _verdict(
    contraction: float,
    overlap: float,
    c_strong: float,
    c_weak: float,
    ovl_dominated: float,
) -> str:
    if not np.isfinite(contraction):
        return "undetermined"
    overlap_high = (not np.isfinite(overlap)) or overlap > ovl_dominated
    if contraction < c_weak and overlap_high:
        return "prior-dominated"
    if contraction >= c_strong:
        return "strong"
    if contraction >= c_weak:
        return "moderate"
    return "weak"  # little narrowing but moved enough to not be prior-dominated


def parameter_learning(
    prior: Any,
    posterior: Any,
    var_names: Sequence[str] | None = None,
    *,
    bins: int = _DEFAULT_BINS,
    c_strong: float = 0.5,
    c_weak: float = 0.1,
    ovl_dominated: float = 0.85,
) -> pd.DataFrame:
    """How much did the data teach us about each parameter, beyond the prior?

    Parameters
    ----------
    prior, posterior:
        Sample containers. Each may be an arviz ``InferenceData`` (the ``"prior"`` /
        ``"posterior"`` group is read), an xarray ``Dataset``, or a ``dict`` mapping a
        parameter name to a sample array (sample axis first). Both must describe the
        same parameters; vector/matrix parameters are compared element-wise.
    var_names:
        Optional restriction. Names match either a flattened element (``"beta[0]"``)
        or a base parameter (``"beta"`` keeps every ``"beta[...]"`` element). ``None``
        compares every parameter present in **both** containers.
    bins:
        Histogram resolution for the overlap coefficient.
    c_strong, c_weak, ovl_dominated:
        Heuristic verdict thresholds (conventions, not law): ``contraction >=
        c_strong`` -> ``"strong"``; ``contraction >= c_weak`` -> ``"moderate"``;
        ``contraction < c_weak`` **and** ``overlap > ovl_dominated`` ->
        ``"prior-dominated"``; otherwise ``"weak"``.

    Returns
    -------
    pandas.DataFrame
        One row per parameter, sorted by ``contraction`` ascending (so the least-learned,
        most prior-dominated parameters sort to the top). Columns: ``parameter``,
        ``prior_mean``, ``prior_sd``, ``post_mean``, ``post_sd``, ``contraction``,
        ``overlap``, ``shift_z``, ``verdict``.

    Notes
    -----
    ``contraction`` is intentionally **not** clipped: a negative value (posterior wider
    than prior) is a genuine warning sign (prior-data conflict or sampling trouble), not
    noise to hide.

    **Informativeness, not importance.** High ``contraction`` / low ``overlap`` means the
    data was *informative* about the parameter -- which **includes confidently pinning it
    near zero**. It is **not** a statement about effect size or sign. Always read the
    posterior ``post_mean`` / interval alongside the diagnostic: contraction tells you
    *the data spoke*; the posterior location tells you *what it said*. (A sign-constrained
    prior such as ``psi = -HalfNormal`` makes "the posterior is below zero" near-automatic;
    a high contraction to a value of ~0 means the data confidently found a *negligible*
    effect, not a confirmed one.)
    """
    pri = _extract(prior, "prior")
    pos = _extract(posterior, "posterior")
    names = [n for n in pos if n in pri]

    if var_names is not None:
        keep = set(var_names)
        names = [n for n in names if n in keep or n.split("[")[0] in keep]

    rows: list[dict[str, Any]] = []
    for name in names:
        a = pri[name][np.isfinite(pri[name])]
        b = pos[name][np.isfinite(pos[name])]
        if a.size == 0 or b.size == 0:
            continue
        prior_sd = float(a.std())
        post_sd = float(b.std())
        prior_mean = float(a.mean())
        post_mean = float(b.mean())
        if prior_sd > 0:
            contraction = 1.0 - (post_sd**2) / (prior_sd**2)
            shift_z = (post_mean - prior_mean) / prior_sd
        else:
            contraction = float("nan")
            shift_z = float("nan")
        overlap = _overlap_coefficient(a, b, bins)
        rows.append(
            {
                "parameter": name,
                "prior_mean": prior_mean,
                "prior_sd": prior_sd,
                "post_mean": post_mean,
                "post_sd": post_sd,
                "contraction": contraction,
                "overlap": overlap,
                "shift_z": shift_z,
                "verdict": _verdict(contraction, overlap, c_strong, c_weak, ovl_dominated),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "parameter", "prior_mean", "prior_sd", "post_mean", "post_sd",
            "contraction", "overlap", "shift_z", "verdict",
        ],
    )
    if not df.empty:
        df = df.sort_values("contraction", na_position="last").reset_index(drop=True)
    return df


# =============================================================================
# Plotting helpers
# =============================================================================


def plot_parameter_learning(
    learning: pd.DataFrame,
    ax: "Axes | None" = None,
    *,
    top: int | None = None,
    metric: str = "contraction",
    threshold: float | None = None,
):
    """Horizontal bar chart of per-parameter learning, colored by ``verdict``.

    ``metric`` is the column to plot (``"contraction"`` by default; ``"overlap"`` also
    works). ``top`` keeps only the ``top`` least-learned parameters (the head of the
    sorted frame). ``threshold`` draws a reference line.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.4 * max(len(learning), 3) + 1.2))

    df = learning if top is None else learning.head(top)
    df = df.iloc[::-1]  # largest at top of the (inverted) horizontal axis
    colors = [_VERDICT_COLORS.get(v, "#8a8079") for v in df["verdict"]]
    ax.barh(df["parameter"], df[metric], color=colors, alpha=0.9)
    ax.set_xlabel(metric)
    ax.axvline(0, color="#2b2118", lw=0.8)
    if threshold is not None:
        ax.axvline(threshold, color="#2b2118", ls="--", lw=1, alpha=0.6)
    ax.set_title(f"Prior -> posterior learning ({metric})")
    # Legend mapping verdict -> color, only for verdicts present.
    from matplotlib.patches import Patch

    present = [v for v in _VERDICT_COLORS if v in set(df["verdict"])]
    if present:
        ax.legend(
            handles=[Patch(color=_VERDICT_COLORS[v], label=v) for v in present],
            loc="lower right", fontsize=8, frameon=False,
        )
    return ax


def plot_prior_posterior_overlay(
    prior: Any,
    posterior: Any,
    parameter: str,
    ax: "Axes | None" = None,
    *,
    bins: int = _DEFAULT_BINS,
    transform=None,
):
    """Overlay the prior and posterior sample histograms for a single parameter.

    The clearest way to *see* whether the data moved/narrowed a parameter relative to
    its prior. ``transform`` (callable) is applied to both sample sets before plotting
    -- e.g. ``lambda x: -x`` to view the signed cannibalization effect from its
    ``psi_..._raw`` HalfNormal parameter.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.6))

    pri = _extract(prior, "prior")
    pos = _extract(posterior, "posterior")
    if parameter not in pri or parameter not in pos:
        raise KeyError(
            f"{parameter!r} not found in both groups. Available (sample): "
            f"{list(pri)[:8]} ..."
        )
    a, b = pri[parameter], pos[parameter]
    if transform is not None:
        a, b = transform(a), transform(b)

    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(a, bins=edges, density=True, color="#8a8079", alpha=0.55, label="prior")
    ax.hist(b, bins=edges, density=True, color="#b5651d", alpha=0.7, label="posterior")
    ax.axvline(float(a.mean()), color="#8a8079", ls="--", lw=1)
    ax.axvline(float(b.mean()), color="#b5651d", ls="--", lw=1.4)
    ax.set_title(f"Prior vs posterior: {parameter}")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False)
    return ax
