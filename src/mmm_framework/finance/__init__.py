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

Lean-core: pydantic and the standard library only. No numpy, no reporting, no
web, no LLM — planning, agents and the server all import it.
"""

from __future__ import annotations

from .evidence import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    EvidenceReference,
    classify_evidence,
    is_cost_kind,
    is_ratio_kind,
    resolve_reference,
)
from .valuation import (
    KpiKind,
    KpiValuation,
    ResolvedValue,
    UnresolvedValueError,
    kpi_to_dollars,
)

__all__ = [
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "EvidenceReference",
    "KpiKind",
    "KpiValuation",
    "ResolvedValue",
    "UnresolvedValueError",
    "classify_evidence",
    "is_cost_kind",
    "is_ratio_kind",
    "kpi_to_dollars",
    "resolve_reference",
]
