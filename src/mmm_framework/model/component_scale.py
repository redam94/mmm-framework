"""The scale convention for registered component Deterministics.

The two model families register the *same* Deterministic names on *different*
scales, and nothing said so:

* the **core** graph (:class:`~mmm_framework.model.base.BayesianMMM`) registers
  ``trend_component``, ``seasonality_component``, ``controls_total`` and
  ``media_total`` on the **standardized** outcome scale, so a consumer bridges
  to KPI units by multiplying by ``y_std``;
* the **extension** graphs (:class:`~mmm_framework.mmm_extensions.models.base.BaseExtendedMMM`
  and its subclasses) register them already multiplied by ``y_std`` — i.e. in
  **original KPI units** — because their own consumers read them directly.

That was discoverable only by reading both graph builders, and the pre-fit
Model Design Readout applied the core convention to both: on a nested model it
multiplied an original-unit component by ``y_std`` a second time, rendering a
prior band scaled by ``y_std**2``. Measured on a 60-week nested model with
``y_std = 68.8`` and a KPI topping out at 1,646: the readout's seasonality band
reached **3,940** — 2.4x the entire KPI — which reads as "the prior is
uninformative", the exact opposite of what a pre-registration document is for.

So the convention now lives here, once, and each family *declares* which side
it is on via ``COMPONENT_DETERMINISTIC_SCALE``. A model that declares nothing —
a duck-typed stub, a third-party subclass — is treated as ``STANDARDIZED``,
which is the historical default and keeps every existing consumer byte-identical.

Consumers should call :func:`to_kpi_units` rather than multiplying by ``y_std``
themselves.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "ComponentScale",
    "COMPONENT_DETERMINISTICS",
    "component_scale",
    "to_kpi_units",
]


class ComponentScale(str, Enum):
    """Which scale a graph registers its component Deterministics on."""

    #: Standardized outcome scale; multiply by ``y_std`` for KPI units.
    STANDARDIZED = "standardized"
    #: Original KPI units; already bridged by the graph.
    ORIGINAL = "original"


#: The Deterministic names this convention governs. These are the additive
#: terms of the outcome mean that both families register under the same names.
#: ``channel_contributions`` is deliberately included: the core graph registers
#: it standardized and the extension graphs register it in KPI units, exactly
#: like the four above.
COMPONENT_DETERMINISTICS = (
    "trend_component",
    "seasonality_component",
    "controls_total",
    "media_total",
    "channel_contributions",
)


def component_scale(model: Any) -> ComponentScale:
    """The scale ``model``'s registered component Deterministics are on.

    Falls back to :attr:`ComponentScale.STANDARDIZED` — the historical
    assumption — for any object that does not declare one, so duck-typed test
    doubles and third-party subclasses keep behaving as they did.
    """
    declared = getattr(model, "COMPONENT_DETERMINISTIC_SCALE", None)
    if declared is None:
        return ComponentScale.STANDARDIZED
    try:
        return ComponentScale(declared)
    except ValueError:  # pragma: no cover - a nonsense declaration
        return ComponentScale.STANDARDIZED


def to_kpi_units(values: np.ndarray, model: Any) -> np.ndarray:
    """Bridge component-Deterministic ``values`` to original KPI units.

    A no-op when the model's graph already did it. Note this bridges a
    *contribution* (a deviation), so it applies ``y_std`` only — never
    ``y_mean``, which belongs to the intercept.
    """
    arr = np.asarray(values, dtype=float)
    if component_scale(model) is ComponentScale.ORIGINAL:
        return arr
    return arr * float(getattr(model, "y_std", 1.0) or 1.0)
