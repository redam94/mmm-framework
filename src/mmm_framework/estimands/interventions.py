"""Registry for :class:`~mmm_framework.estimands.spec.CustomIntervention`.

A custom intervention names a registered callable (``ref``) that maps the
factual media matrix to a counterfactual one. The registry is empty by default;
a model family or notebook registers its own (e.g. a flighting schedule, a
competitor-response scenario) via :func:`register_intervention`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

#: ref -> callable(model, params, X) -> X'  (X is a raw-scale media matrix copy)
_INTERVENTIONS: dict[str, Callable[[Any, dict[str, Any], np.ndarray], np.ndarray]] = {}


def register_intervention(
    ref: str, fn: Callable[[Any, dict[str, Any], np.ndarray], np.ndarray]
) -> None:
    """Register a custom-intervention callable under ``ref``."""
    _INTERVENTIONS[ref] = fn


def apply_custom_intervention(
    model: Any, intervention: Any, X: np.ndarray
) -> np.ndarray:
    """Apply a registered custom intervention to ``X`` (a raw media-matrix copy)."""
    ref = intervention.ref
    fn = _INTERVENTIONS.get(ref)
    if fn is None:
        raise ValueError(
            f"Unknown custom intervention ref {ref!r}; register it with "
            "mmm_framework.estimands.interventions.register_intervention."
        )
    return fn(model, dict(intervention.params), X)


__all__ = ["register_intervention", "apply_custom_intervention"]
