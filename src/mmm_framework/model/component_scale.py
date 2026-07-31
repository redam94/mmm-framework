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
    "kpi_scale_bridge_reason",
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
    """Undo the standardization a component Deterministic was registered under.

    A no-op when the model's graph already did it. This bridges a *contribution*
    (a deviation), so it applies ``y_std`` only — never ``y_mean``, which
    belongs to the intercept.

    **It is the standardization bridge, not a link inverse.** On an additive,
    Gaussian model the result is in original KPI units, which is the usual case
    and the one every caller here is written for. On a model whose outcome is
    not the KPI itself the result is on that model's own scale: a
    ``MULTIPLICATIVE`` specification standardizes ``log(y)``, so the result is
    log-scale (use ``compute_component_decomposition``'s LMDI index to reach KPI
    units), and a count/bounded likelihood sets ``y_std = 1.0``, so the result is
    on the link scale. Neither is expressible in a per-family constant, and
    neither is changed by this function — it is named here so a caller does not
    read "KPI units" into a number that is not.
    """
    arr = np.asarray(values, dtype=float)
    if component_scale(model) is ComponentScale.ORIGINAL:
        return arr
    return arr * float(getattr(model, "y_std", 1.0) or 1.0)


#: Likelihood families whose outcome — and therefore whose in-graph components
#: — live on a LINK scale rather than the KPI's. Listed explicitly, so an
#: unrecognized or absent family is treated as the historical Gaussian default
#: rather than refused.
_LINK_SCALE_FAMILIES = frozenset(
    {
        "lognormal",
        "binomial",
        "beta_binomial",
        "poisson",
        "negative_binomial",
        "beta",
    }
)

#: Non-identity links, for a model that declares one without a listed family.
_NON_IDENTITY_LINKS = frozenset({"logit", "log"})


def kpi_scale_bridge_reason(model: Any) -> str | None:
    """Why :func:`to_kpi_units` would NOT reach KPI units for ``model``.

    Returns ``None`` when the bridge is genuine — the usual additive, Gaussian
    case — and otherwise a phrase naming the reason, for a caller to put in a
    refusal.

    There are **two** ways the bridge falls short, and they are stated together
    in this module's own docstring because guarding only one is the natural
    mistake:

    * a ``MULTIPLICATIVE`` specification standardizes ``log(y)``, so the
      component is on the log scale;
    * a non-Gaussian likelihood or a non-identity link keeps the outcome on the
      link scale and sets ``y_std = 1.0``, so ``to_kpi_units`` is the identity
      over (for example) logits.

    Measured on the binomial awareness garden model, whose contributions are
    logit-scale goodwill: an unguarded ``contribution_roi`` published 0.0071 for
    a channel whose original-scale ROI-equivalent is 1.57 — over 200x too small,
    with ``status="ok"`` and ``units="ROI"``.
    """
    # Refuse only on a KNOWN-bad configuration, never on an unrecognized one.
    # `_get_contribution_samples` is deliberately tolerant of duck-typed models,
    # and `getattr(mock, "_multiplicative", False)` on a `Mock` returns a truthy
    # `Mock` — so a loose predicate refuses every test double and every
    # third-party object, which is the over-broad direction this guard exists to
    # avoid. Hence `is True` and an explicit family list.
    if getattr(model, "_multiplicative", False) is True:
        return (
            "the model uses the multiplicative (semi-log) specification, so the "
            "in-graph components are on the LOG scale"
        )

    likelihood = getattr(getattr(model, "model_config", None), "likelihood", None)
    family = getattr(getattr(likelihood, "family", None), "value", None)
    link = getattr(getattr(likelihood, "link", None), "value", None)
    if isinstance(family, str) and family in _LINK_SCALE_FAMILIES:
        return (
            f"the model has a {family!r} likelihood, so the outcome — and the "
            "in-graph components with it — are on the LINK scale, not the KPI's"
        )
    if isinstance(link, str) and link in _NON_IDENTITY_LINKS:
        return (
            f"the model uses a {link!r} link, so the in-graph components are on "
            "the link scale, not the KPI's"
        )
    return None
