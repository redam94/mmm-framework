"""Graph fingerprint for BayesianMMM-family models (src-refactor PR 0.1).

Structure-only fingerprinting produces a FALSE NEGATIVE: ``michaelis_menten``
and ``tanh`` saturation collide on a structural hash — both emit
``sat_half_<ch>`` with the same Beta prior, and only the Deterministic formula
differs. The numeric block (Deterministic values + per-factor logp at a fixed
probe point) is therefore mandatory, not optional. Any refactor of the
saturation/adstock/trend builders could otherwise swap two families past a
structural test.

Four gotchas this implementation must keep (each was found by running it —
see ``technical-docs/src-refactor.md`` §PR 0.1):

1. ``initial_point()`` is degenerate: several components evaluate to exactly
   0.0 there. Every transformed-space entry is offset before probing.
2. The offset is NAME-derived (a stable hash of the variable name), never
   position-indexed — a positional offset re-randomises every alphabetically
   later variable when an RV is added or renamed, destroying the readable
   diff the goldens exist for.
3. The offset cannot rescue zero-mean blocks under ``sum`` (a whole number of
   Fourier cycles sums to 0 for ANY coefficient) — the ``abs-sum`` second
   element of ``deterministic_values`` is load-bearing.
4. ``replace_rvs_by_values`` is mandatory when compiling the Deterministics:
   a naive ``compile_fn`` still contains RandomVariables and re-samples them
   per process. ``inputs=model.value_vars`` is likewise required.

Store full dicts, not hashes: a failure must be a readable diff.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _round(x: float, nd: int = 9) -> float:
    v = float(np.round(float(x), nd))
    return 0.0 if v == 0.0 else v  # normalize -0.0


def _name_offset(name: str, scale: float = 0.1) -> float:
    """Stable, name-derived probe offset in (0, scale] (gotcha 2)."""
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return scale * (1 + (h % 97)) / 97.0


def _shape(var: Any) -> list[int]:
    try:
        return [int(s) for s in var.shape.eval()]
    except Exception:  # noqa: BLE001 — symbolic-only shape
        return list(getattr(var.type, "shape", ()) or [])


def _dist_params(rv: Any) -> list[str]:
    """The RV's non-rng op inputs, rendered stably (constants as values)."""
    out: list[str] = []
    for inp in rv.owner.inputs:
        t = getattr(inp, "type", None)
        tname = type(t).__name__ if t is not None else type(inp).__name__
        if tname == "RandomGeneratorType" or "RandomState" in tname:
            continue
        try:
            data = inp.data  # pytensor Constant
            out.append(repr(np.round(np.asarray(data, dtype=float), 9).tolist()))
        except Exception:  # noqa: BLE001 — a graph input, not a constant
            out.append(getattr(inp, "name", None) or tname)
    return out


def model_fingerprint(model: Any, *, numeric: bool = True) -> dict[str, Any]:
    """A structural + numeric fingerprint of a built (unfitted) PyMC model.

    ``model`` is a ``BayesianMMM``-family instance exposing ``.model`` (the
    ``pm.Model``), or a bare ``pm.Model``.
    """
    import pymc as pm

    m = getattr(model, "model", model)
    assert isinstance(m, pm.Model), f"expected a pm.Model, got {type(m)}"

    fp: dict[str, Any] = {
        "free_RVs": sorted(
            [
                rv.name,
                rv.owner.op.name or type(rv.owner.op).__name__,
                _shape(rv),
                _dist_params(rv),
                type(m.rvs_to_transforms.get(rv)).__name__,  # None -> "NoneType"
            ]
            for rv in m.free_RVs
        ),
        "observed_RVs": sorted(
            [
                rv.name,
                rv.owner.op.name or type(rv.owner.op).__name__,
                _shape(rv),
            ]
            for rv in m.observed_RVs
        ),
        "deterministics": sorted([d.name, _shape(d)] for d in m.deterministics),
        "potentials": sorted(p.name for p in m.potentials),
        "data_vars": sorted(
            [name, list(np.shape(var.get_value()))]
            for name, var in m.named_vars.items()
            if hasattr(var, "get_value") and var not in m.deterministics
        ),
        "coords": {str(k): [str(x) for x in v] for k, v in sorted(m.coords.items())},
        "dims": {
            str(k): [str(d) for d in v] for k, v in sorted(m.named_vars_to_dims.items())
        },
    }
    if not numeric:
        return fp

    # ── the probe point: initial_point, offset per NAME (gotchas 1 + 2) ────
    ip = m.initial_point()
    probe = {
        name: np.asarray(val, dtype=float) + _name_offset(name)
        for name, val in ip.items()
    }
    fp["initial_point"] = {
        name: [list(np.shape(val)), _round(np.sum(val))]
        for name, val in sorted(ip.items())
    }

    # ── per-factor logp at the probe point ────────────────────────────────
    factors = m.basic_RVs + m.potentials
    logp_fn = m.compile_fn(m.logp(vars=factors, sum=False), inputs=m.value_vars)
    logps = logp_fn(probe)
    fp["logp_terms"] = {
        f.name: _round(np.sum(v))
        for f, v in sorted(zip(factors, logps), key=lambda t: t[0].name)
    }
    fp["total_logp"] = _round(sum(np.sum(v) for v in logps))

    # ── Deterministic values at the probe point (gotchas 3 + 4) ───────────
    if m.deterministics:
        outs = m.replace_rvs_by_values(list(m.deterministics))
        # inputs=value_vars is required (PointFunc otherwise raises on the
        # first value var a Deterministic does not depend on) AND unused
        # inputs must be tolerated for exactly the same reason.
        det_fn = m.compile_fn(outs, inputs=m.value_vars, on_unused_input="ignore")
        vals = det_fn(probe)
        fp["deterministic_values"] = {
            d.name: [_round(np.sum(v)), _round(np.sum(np.abs(v)))]
            for d, v in sorted(zip(m.deterministics, vals), key=lambda t: t[0].name)
        }
    else:
        fp["deterministic_values"] = {}
    return fp


def fingerprint_digest(fp: dict[str, Any]) -> str:
    """A stable digest for quick comparisons; the DICT is the contract."""
    import json

    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
