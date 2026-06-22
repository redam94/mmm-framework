"""Model capability detection for estimand gating.

:func:`model_capabilities` returns a ``set[str]`` of capability flags by **cheap
duck-typing** — no graph build, no posterior-predictive sampling. An
:class:`~mmm_framework.estimands.spec.Estimand` declares
``required_capabilities``; the evaluator returns an ``unsupported`` result when
they are not met (so a non-MMM model auto-filters MMM-only estimands rather than
crashing).

Capabilities are plain strings; parameterized ones use ``"NS:arg"`` (e.g.
``"HAS_LATENT:awareness"``) so a typed need — a future CFA's
``"HAS_FACTOR_LOADINGS"`` — is gateable without a schema change.
"""

from __future__ import annotations

from typing import Any

# Canonical flag names (string constants so callers don't typo them).
HAS_CHANNELS = "HAS_CHANNELS"
HAS_CONTRIBUTIONS = "HAS_CONTRIBUTIONS"
HAS_CONTRIBUTION_DETERMINISTIC = "HAS_CONTRIBUTION_DETERMINISTIC"
IN_GRAPH_RESPONSE_CURVE = "IN_GRAPH_RESPONSE_CURVE"
HAS_LATENT = "HAS_LATENT"  # parameterized: f"HAS_LATENT:{name}"


def _posterior(model: Any) -> Any | None:
    """The posterior group, or None when the model is unfitted/traceless."""
    trace = getattr(model, "_trace", None) or getattr(model, "trace", None)
    if trace is None:
        return None
    return getattr(trace, "posterior", None)


def model_capabilities(model: Any) -> set[str]:
    """Capability flags for ``model`` (safe on an unfitted model).

    Parameters
    ----------
    model:
        Any object duck-typing the MMM surface (``channel_names``, optionally a
        fitted ``_trace``).
    """
    caps: set[str] = set()

    channels = getattr(model, "channel_names", None)
    if channels:
        caps.add(HAS_CHANNELS)
        if hasattr(model, "compute_counterfactual_contributions"):
            caps.add(HAS_CONTRIBUTIONS)

    if getattr(model, "use_parametric_adstock", False):
        caps.add(IN_GRAPH_RESPONSE_CURVE)

    posterior = _posterior(model)
    if posterior is not None:
        try:
            var_names = set(posterior.data_vars)  # xarray Dataset
        except Exception:  # noqa: BLE001
            var_names = set()

        if "channel_contributions" in var_names or any(
            v.startswith(("contribution_", "channel_contribution_")) for v in var_names
        ):
            caps.add(HAS_CONTRIBUTION_DETERMINISTIC)

        # Expose every posterior data var as a gateable latent so a declared
        # estimand can require e.g. "HAS_LATENT:awareness".
        for v in var_names:
            caps.add(f"{HAS_LATENT}:{v}")

    return caps


def missing_capabilities(required: list[str], available: set[str]) -> list[str]:
    """The subset of ``required`` not present in ``available`` (order-preserving)."""
    return [c for c in required if c not in available]


__all__ = [
    "HAS_CHANNELS",
    "HAS_CONTRIBUTIONS",
    "HAS_CONTRIBUTION_DETERMINISTIC",
    "IN_GRAPH_RESPONSE_CURVE",
    "HAS_LATENT",
    "model_capabilities",
    "missing_capabilities",
]
