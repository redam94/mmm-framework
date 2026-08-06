"""Money. What one KPI unit is worth, and where that number came from.

Every planning surface that recommends a *dollar* amount divides by, or
multiplies through, a value-per-KPI-unit. Before v1.4 that quantity was
expressed at least six different ways across the codebase, each with its own
``= 1.0`` default, and the default was producing a wrong number in production:
the Planner's "Fund to breakeven" control never sent one, the server filled in
``1.0``, and the free-mode objective then funded every channel until the
marginal return was one KPI unit per dollar. On a KPI denominated in thousands
that recommendation is off by ~1000x — rendered with credible intervals.

This package is the single answer. :func:`kpi_to_dollars` resolves the value
once, through one precedence chain, and returns it with its ``source`` attached
so every consumer can say where it came from. **Unresolved is a first-class
state, not 1.0** — see :class:`ResolvedValue`.

The name is not new: ``technical-docs/experiment-net-economics.md`` §79 already
specified ``kpi_to_dollars(EVOI, margin_per_kpi | price, kpi_kind)`` and it was
never implemented. This implements that, rather than minting a seventh spelling.

A second question lives here for the same reason. :mod:`~.lines` says where a
number on a bridge came from, and :mod:`~.closure` reconciles a fit's components
against the observed KPI so the residual gets disclosed rather than absorbed
into a bar labelled "base demand" (issue #220). Money and closure share a
package because a P&L rollup needs both and should not import two stacks.

Lean-core: pydantic and the standard library only. No numpy, no reporting, no
web, no LLM — planning, agents and the server all import it. ``closure`` is the
one exception; it needs numpy and a fitted model, so it is imported lazily and
only when something actually asks for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .evidence import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    EvidenceReference,
    classify_evidence,
    is_cost_kind,
    is_ratio_kind,
    resolve_reference,
)
from .lines import (
    ABSORBING,
    MODELLED,
    OBSERVED,
    RESIDUAL,
    SUPPLIED,
    BridgeLine,
    LineProvenance,
    absorbs_residual,
    bridge_gap,
    provenance_of,
)
from .valuation import (
    KpiKind,
    KpiValuation,
    ResolvedValue,
    UnresolvedValueError,
    kpi_to_dollars,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .closure import ClosureFacts, MediaReconciliation, decomposition_closure

#: Names served from :mod:`.closure`, which pulls numpy. Kept out of the eager
#: import list so ``from mmm_framework.finance import kpi_to_dollars`` stays as
#: cheap as it was.
_LAZY = {
    "ClosureFacts": "closure",
    "MediaReconciliation": "closure",
    "decomposition_closure": "closure",
    "fitted_total": "closure",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy export for the closure module."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


__all__ = [
    "ABSORBING",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "MODELLED",
    "OBSERVED",
    "RESIDUAL",
    "SUPPLIED",
    "BridgeLine",
    "ClosureFacts",
    "EvidenceReference",
    "KpiKind",
    "KpiValuation",
    "LineProvenance",
    "MediaReconciliation",
    "ResolvedValue",
    "UnresolvedValueError",
    "absorbs_residual",
    "bridge_gap",
    "classify_evidence",
    "decomposition_closure",
    "fitted_total",
    "is_cost_kind",
    "is_ratio_kind",
    "kpi_to_dollars",
    "provenance_of",
    "resolve_reference",
]
