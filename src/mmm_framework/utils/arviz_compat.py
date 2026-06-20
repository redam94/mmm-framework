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


def point_to_idata(point: dict):
    """Wrap a ``find_MAP`` point dict into a (chain=1, draw=1) InferenceData.

    Drops PyTensor's transformed duplicates (``*_log__`` / ``*_interval__``) and
    keeps the constrained values plus deterministics, matching the variable names
    produced by NUTS.

    arviz flipped ``from_dict``'s calling convention between the legacy
    ``InferenceData`` and the ``DataTree`` era — and the *wrong* form does not
    raise: it silently wraps everything as one variable named ``"posterior"``. So
    we try both conventions and keep whichever actually materialized the
    variables; if neither does, build the dataset by hand.
    """
    import arviz as az

    posterior = {
        name: np.asarray(value)[np.newaxis, np.newaxis, ...]
        for name, value in point.items()
        if not name.endswith("__")
    }
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
    ds = xr.Dataset(data_vars, coords={"chain": [0], "draw": [0]})
    try:
        return az.InferenceData(posterior=ds)
    except Exception:  # noqa: BLE001 - DataTree-era arviz
        dt = xr.DataTree()
        dt["posterior"] = ds
        return dt
