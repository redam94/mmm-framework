"""Version-robust shims for arviz / pymc API drift.

The framework targets a range of arviz and pymc versions whose public APIs
shifted in ways that fail *loudly* (``TypeError``/``AttributeError``) or, worse,
*silently* (wrong result). Centralizing the shims here keeps every call site
honest and avoids re-discovering the same breakage module by module.

Covered drift:
- ``pm.sample_prior_predictive`` renamed ``samples`` -> ``draws`` (pymc >=5.x).
- arviz containers migrated ``InferenceData`` -> xarray ``DataTree`` (arviz
  >=0.22): ``.groups`` became a property of slash-prefixed paths (not a method),
  ``.extend`` is gone, and ``Dataset.to_array`` became ``to_dataarray``.
- ``az.from_dict`` flipped its calling convention across that migration AND the
  wrong form fails silently (wraps everything as one var ``"posterior"``).
- ``az.hdi`` renamed ``hdi_prob`` -> ``prob`` (arviz >=1.x) AND changed 2-d
  ndarray semantics from pooled ``(chain, draw)`` to a batch of independent
  rows — use :func:`hdi_bounds` for the historical pooled interval.
"""

from __future__ import annotations

import warnings

import numpy as np


def sample_prior_predictive(samples: int, random_seed: int | None = None):
    """Call ``pm.sample_prior_predictive`` across the ``samples``->``draws`` rename.

    Must be called inside a ``with model:`` context. Callers keep their public
    ``samples`` argument.
    """
    import pymc as pm

    try:
        return pm.sample_prior_predictive(draws=samples, random_seed=random_seed)
    except TypeError:  # pragma: no cover - legacy pymc fallback
        return pm.sample_prior_predictive(samples=samples, random_seed=random_seed)


def dataset_extremum(ds, kind: str) -> float:
    """Reduce an arviz per-variable stats container to one scalar.

    ``az.rhat`` / ``az.ess`` return an xarray ``Dataset`` on legacy arviz and a
    ``DataTree`` on arviz >=0.22; both expose ``data_vars``, so we reduce over
    those rather than relying on the renamed ``to_array``/``to_dataarray``.
    All-NaN variables (e.g. R-hat for a single-chain fit) are skipped so the
    reduction stays quiet and meaningful. ``kind`` is ``"max"`` or ``"min"``.
    """
    reduce = np.nanmax if kind == "max" else np.nanmin
    values = []
    for var in ds.data_vars.values():
        arr = np.asarray(var.values)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        values.append(reduce(arr))
    if not values:
        return float("nan")
    return float(reduce(values))


def hdi_bounds(samples, prob: float) -> tuple[float, float]:
    """``(lo, hi)`` highest-density interval of the POOLED ``samples``.

    Absorbs two arviz 1.x changes at once: ``hdi_prob`` was renamed ``prob``,
    and a 2-d ndarray input is now treated as a BATCH of independent rows (one
    interval per row) instead of pooled ``(chain, draw)`` draws. Inputs are
    flattened here so every call site keeps the historical pooled semantics.
    """
    import arviz as az

    flat = np.asarray(samples, dtype=float).ravel()
    try:
        res = az.hdi(flat, prob=prob)
    except TypeError:  # pragma: no cover - legacy arviz (<1.x) fallback
        res = az.hdi(flat, hdi_prob=prob)
    arr = np.asarray(res).ravel()
    return float(arr[0]), float(arr[1])


def group_names(idata) -> list[str]:
    """Return an arviz container's group names, normalized (no leading ``/``).

    Legacy ``InferenceData`` exposes ``groups()`` as a method returning bare
    names; the newer ``DataTree`` exposes ``groups`` as a property of
    slash-prefixed paths. Both are normalized to bare names here.
    """
    groups = getattr(idata, "groups", None)
    raw = groups() if callable(groups) else (groups or [])
    return [str(g).lstrip("/") for g in raw if str(g).lstrip("/")]


def has_group(idata, name: str) -> bool:
    """True if an arviz container exposes the named group (robust across APIs)."""
    return name in group_names(idata)


def attach_prior(trace, prior):
    """Merge the prior groups into the posterior trace, best-effort.

    ``InferenceData.extend`` exists on the legacy container but not on the newer
    ``DataTree``. Prior groups are only used for prior-vs-posterior tooling, so
    on any incompatibility we warn and return the trace unchanged.
    """
    extend = getattr(trace, "extend", None)
    if callable(extend):
        try:
            extend(prior)
            return trace
        except Exception:  # noqa: BLE001
            pass
    try:
        names = group_names(prior)
        for group in ("prior", "prior_predictive"):
            if group in names:
                trace[group] = prior[group]
        return trace
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not attach prior groups to the trace ({exc}); "
            "prior-vs-posterior diagnostics will be unavailable.",
            stacklevel=2,
        )
        return trace


def posterior_from_dict(posterior: dict):
    """Build an arviz container whose ``posterior`` group holds ``posterior``.

    Values must already be shaped ``(chain, draw, *shape)``. arviz flipped
    ``from_dict``'s calling convention between the legacy ``InferenceData`` and
    the ``DataTree`` era — and the *wrong* form does not raise: it silently
    wraps everything as one variable named ``"posterior"``. So we try both
    conventions and keep whichever actually materialized the variables; if
    neither does, build the dataset by hand.
    """
    import arviz as az

    posterior = {name: np.asarray(value) for name, value in posterior.items()}
    expected = set(posterior)

    for build in (
        lambda: az.from_dict(posterior=posterior),  # legacy / arviz<=0.22
        lambda: az.from_dict({"posterior": posterior}),  # DataTree era
    ):
        # The wrong convention may raise OR succeed-but-malformed (no usable
        # ``.posterior`` group), so validate inside the guard and treat any
        # failure as "try the next form" — never raise out of the loop.
        try:
            idata = build()
            data_vars = set(idata.posterior.data_vars)
        except Exception:  # noqa: BLE001
            continue
        if expected and expected.issubset(data_vars):
            return idata

    import xarray as xr

    data_vars = {}
    for name, arr in posterior.items():
        extra = [f"{name}_dim_{i}" for i in range(arr.ndim - 2)]
        data_vars[name] = (["chain", "draw", *extra], arr)
    n_chains = next(iter(posterior.values())).shape[0] if posterior else 1
    n_draws = next(iter(posterior.values())).shape[1] if posterior else 1
    ds = xr.Dataset(
        data_vars,
        coords={"chain": list(range(n_chains)), "draw": list(range(n_draws))},
    )
    return dataset_to_idata(ds)


def summary(data, var_names=None, **kwargs):
    """``az.summary`` with NUMERIC values across the arviz 1.x formatting change.

    arviz 1.x defaults ``round_to="auto"``, which returns formatted STRINGS
    (object dtype) — silent poison for numeric consumers: ``max()`` on the
    ``r_hat`` column becomes a LEXICOGRAPHIC string max ("9.99" > "10.01").
    ``round_to="none"`` restores raw floats on both arviz lines. Note the
    interval columns also drifted (``hdi_3%``/``hdi_97%`` → ``eti89_lb``/
    ``eti89_ub``, an 89% equal-tailed default) — pass ``ci_prob``/``ci_kind``
    explicitly if you consume them.
    """
    import arviz as az

    kwargs.setdefault("round_to", "none")
    return az.summary(data, var_names=var_names, **kwargs)


def hdi_dataset(data, prob: float, var_names=None):
    """Trace-level HDI across the ``hdi_prob`` → ``prob`` rename.

    ``result[var].values`` keeps the legacy contract on both arviz lines:
    shape ``(*var_shape, 2)`` with the last axis = (lower, upper).
    """
    import arviz as az

    try:
        return az.hdi(data, prob=prob, var_names=var_names)
    except TypeError:  # pragma: no cover - legacy arviz (<1.x) fallback
        return az.hdi(data, hdi_prob=prob, var_names=var_names)


def plot_posterior(data, var_names=None, **kwargs):
    """Posterior-distribution plot across the arviz-plots split.

    Legacy arviz exposed ``az.plot_posterior``; arviz 1.x moved plotting into
    ``arviz_plots`` where the equivalent is ``plot_dist`` (defaults to the
    posterior group). ``az.plot_trace`` survived the split (re-exported), so
    only this one needs a shim.
    """
    import arviz as az

    if hasattr(az, "plot_posterior"):  # legacy arviz (<1.x)
        return az.plot_posterior(data, var_names=var_names, **kwargs)
    import arviz_plots as azp

    return azp.plot_dist(data, var_names=var_names, **kwargs)


def dataset_to_idata(ds, group: str = "posterior"):
    """Wrap a ready xarray ``Dataset`` as one group of an arviz container.

    ``az.InferenceData(posterior=ds)`` stopped working on the DataTree
    migration (``DataTree.__init__`` takes no group kwargs); the DataTree era
    assigns groups by key instead. Works on both.
    """
    import arviz as az
    import xarray as xr

    try:
        return az.InferenceData(**{group: ds})
    except Exception:  # noqa: BLE001 - DataTree-era arviz
        dt = xr.DataTree()
        dt[group] = ds
        return dt


def point_to_idata(point: dict):
    """Wrap a ``find_MAP`` point dict into a (chain=1, draw=1) InferenceData.

    Drops PyTensor's transformed duplicates (``*_log__`` / ``*_interval__``) and
    keeps the constrained values plus deterministics, matching the variable names
    produced by NUTS. Container construction (and the ``from_dict`` convention
    flip) is handled by :func:`posterior_from_dict`.
    """
    return posterior_from_dict(
        {
            name: np.asarray(value)[np.newaxis, np.newaxis, ...]
            for name, value in point.items()
            if not name.endswith("__")
        }
    )


def psis_log_weights(log_lik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSIS-smoothed LOO log-weights from pointwise log-likelihood draws.

    ``log_lik`` is ``(n_obs, n_samples)`` (draws on the last axis). Returns
    ``(log_weights, khat)`` with ``log_weights`` the same shape (normalized so
    ``logsumexp == 0`` per observation) and ``khat`` of shape ``(n_obs,)`` —
    the Pareto shape diagnostic (``khat > 0.7`` means the weights for that
    observation are unreliable).

    arviz 1.x removed the top-level ``az.psislw``; the smoother now lives on
    ``arviz_stats.base.array_stats``. The legacy ``az.psislw`` (draws on the
    FIRST axis) is the fallback for pre-1.x environments.
    """
    ll = np.asarray(log_lik, dtype=float)
    if ll.ndim != 2:
        raise ValueError(f"log_lik must be 2-D (n_obs, n_samples), got {ll.shape}")
    try:
        from arviz_stats.base import array_stats

        lw, khat = array_stats.psislw(-ll, r_eff=1.0, axis=-1)
        return np.asarray(lw), np.asarray(khat)
    except ImportError:  # pragma: no cover - legacy arviz (<1.x) fallback
        import arviz as az

        lw, khat = az.psislw(-ll.T)
        return np.asarray(lw).T, np.asarray(khat)
