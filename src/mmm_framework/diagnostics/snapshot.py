"""Fit-time model-health snapshot: the JSON block a ``model_run`` carries so
the UI can show *whether to believe the fit* — not just what it estimated.

Two layers, mirroring the docs' honesty contract (green diagnostics validate
computation, not causality):

* **convergence** — did the sampler work? Divergences, max R-hat, min bulk ESS
  (already computed by ``BayesianMMM.fit``; recomputed from the trace as a
  fallback for models that don't stamp them).
* **learning** — did the *data* inform the parameters, or are the posteriors
  re-stating the priors? Per-parameter prior→posterior contraction / overlap /
  shift verdicts from :func:`mmm_framework.diagnostics.parameter_learning`,
  reusing the prior group already attached to the trace at fit time (no extra
  sampling).

Like ``planning.history.compute_run_metrics``, this runs kernel-side at fit
time, must touch only the fitted model, and is best-effort: callers wrap it in
try/except and a failure never fails a fit.
"""

from __future__ import annotations

import math
from typing import Any

FIT_DIAGNOSTICS_SCHEMA_VERSION = 1

# Conventional sampler-health thresholds (Vehtari et al. 2021 for R-hat; the
# 100-per-chain bulk-ESS rule of thumb at 4 chains).
RHAT_OK = 1.01
ESS_BULK_OK = 400.0


def _f(x: Any) -> float | None:
    """Finite float or None — keeps the snapshot JSON-safe (no NaN/inf)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _convergence_block(mmm: Any, results: Any) -> dict[str, Any]:
    diag = dict(getattr(results, "diagnostics", None) or {})
    if not diag:
        trace = getattr(mmm, "_trace", None)
        if trace is not None:
            import arviz as az

            from ..utils import arviz_compat

            try:
                diag["divergences"] = int(trace.sample_stats.diverging.sum().values)
            except Exception:
                diag["divergences"] = None
            try:
                diag["rhat_max"] = arviz_compat.dataset_extremum(az.rhat(trace), "max")
                diag["ess_bulk_min"] = arviz_compat.dataset_extremum(
                    az.ess(trace, method="bulk"), "min"
                )
            except Exception:
                pass

    divergences = diag.get("divergences")
    rhat_max = _f(diag.get("rhat_max"))
    ess_bulk_min = _f(diag.get("ess_bulk_min"))
    flags: list[str] = []
    if divergences:
        flags.append("divergences")
    if rhat_max is not None and rhat_max > RHAT_OK:
        flags.append("rhat")
    if ess_bulk_min is not None and ess_bulk_min < ESS_BULK_OK:
        flags.append("ess")
    return {
        "divergences": int(divergences) if divergences is not None else None,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "rhat_threshold": RHAT_OK,
        "ess_threshold": ESS_BULK_OK,
        "flags": flags,
        "ok": not flags,
    }


def _learning_block(mmm: Any, max_parameters: int) -> dict[str, Any] | None:
    from ..utils import arviz_compat
    from .learning import parameter_learning

    trace = getattr(mmm, "_trace", None)
    var_names = [rv.name for rv in mmm.model.free_RVs]
    if trace is not None and arviz_compat.has_group(trace, "prior"):
        # fit() extends the trace with a prior draw — reuse it.
        df = parameter_learning(trace, trace, var_names=var_names)
    else:
        df = mmm.compute_parameter_learning(var_names=var_names)
    if df is None or df.empty:
        return None

    counts: dict[str, int] = {}
    for v in df["verdict"]:
        counts[str(v)] = counts.get(str(v), 0) + 1

    # The frame is sorted by contraction ascending, so head() keeps the
    # least-learned (most prior-dominated) parameters — the ones worth showing.
    rows = [
        {
            "parameter": str(r["parameter"]),
            "verdict": str(r["verdict"]),
            "contraction": _f(r["contraction"]),
            "contraction_robust": _f(r["contraction_robust"]),
            "overlap": _f(r["overlap"]),
            "shift_z": _f(r["shift_z"]),
            "post_mean": _f(r["post_mean"]),
            "post_sd": _f(r["post_sd"]),
            "post_ess_bulk": _f(r["post_ess_bulk"]),
        }
        for _, r in df.head(max_parameters).iterrows()
    ]
    return {
        "n_parameters": int(len(df)),
        "verdict_counts": counts,
        "parameters": rows,
        "truncated": len(df) > max_parameters,
    }


def compute_fit_diagnostics(
    mmm: Any, results: Any = None, *, max_parameters: int = 40
) -> dict[str, Any]:
    """JSON-safe model-health snapshot (schema v1) for a fitted model.

    ``results`` is the :class:`~mmm_framework.model.results.MMMResults` whose
    ``diagnostics`` dict already carries divergences/R-hat/ESS; pass None to
    recompute from the trace. Each layer is independently best-effort, so a
    learning failure still yields the convergence block (and vice versa).
    """
    out: dict[str, Any] = {"schema_version": FIT_DIAGNOSTICS_SCHEMA_VERSION}
    try:
        out["convergence"] = _convergence_block(mmm, results)
    except Exception as exc:  # noqa: BLE001
        out["convergence_error"] = str(exc)
    try:
        out["learning"] = _learning_block(mmm, max_parameters)
    except Exception as exc:  # noqa: BLE001
        out["learning_error"] = str(exc)
    return out


__all__ = ["compute_fit_diagnostics", "FIT_DIAGNOSTICS_SCHEMA_VERSION"]
