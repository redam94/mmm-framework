"""Which direction is better for this estimand, and against what reference.

Every estimand table in this codebase grades a number by asking whether its
interval clears a no-effect reference. That rule was written for quantities
where **more is better** — an ROI against 1.0, an incremental contribution
against 0 — and it is applied to every estimand by two mirrored copies of the
same heuristic (``platform/estimands.py::is_ratio_kind`` and
``reporting/sections.py::EstimandsSection._is_ratio_kind``).

A **cost per outcome** breaks it in both halves at once. ``cost_per_conversion``
(kind ``cost_per_outcome``, units ``$/conversion``) is neither an ROI nor a
contribution, so it fell to the ``else`` branch and was graded against ``0.0``
with "higher is better". Executed on the shipped grader before this module
existed::

    $45 CPA, CI [30, 62]  ->  "strong"     # ruinous if a conversion is worth $20
    $2  CPA, CI [ 1,  3]  ->  "strong"     # identical verdict

Both read Strong because a cost is above zero *by construction*. The interval
clearing the reference carried no information at all, and the one channel a
media buyer needed to stop funding was labelled the same as the one to scale.

Two rules follow, and this module is where they live so the report and the
dashboard cannot drift apart again:

* **Direction is a property of the metric.** A cost is graded
  :data:`LOWER_IS_BETTER`: the evidence is strong when the interval lies
  *below* the reference.
* **A cost has no free reference.** Zero is not a break-even — nothing beats
  free. The reference is the value of one outcome, which is exactly the
  quantity :func:`~mmm_framework.finance.valuation.kpi_to_dollars` resolves, and
  when nothing resolves the honest verdict is "not assessable" rather than a
  grade against a fabricated bar. This is the same first-class-unresolved rule
  :mod:`~mmm_framework.finance.valuation` enforces for money.

Lean-core: standard library only. Imported by ``platform`` (no reporting stack)
and by ``reporting`` (no services layer) alike.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "UNRESOLVED_COST_REASON",
    "EvidenceReference",
    "classify_evidence",
    "is_cost_kind",
    "is_ratio_kind",
    "resolve_reference",
    "verdict_label",
]

#: Grading directions. Stored as plain strings so they ride through JSON, the
#: persisted estimand rows and the REST payload without an enum import.
HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"

#: Units strings that mean "this number is a bare ratio" (ROI-like).
_RATIO_UNITS = {"ratio", "x", "multiple"}


def is_ratio_kind(kind: str | None, units: str | None) -> bool:
    """True for ROI / ROAS-style ratios, whose no-effect reference is 1.0."""
    k = (kind or "").lower()
    u = (units or "").lower()
    if is_cost_kind(kind, units):
        # "$/conversion" is a ratio arithmetically and nothing like one for
        # grading; cost wins so a future kind naming both cannot regress.
        return False
    return "roi" in k or "roas" in k or u in _RATIO_UNITS


def is_cost_kind(kind: str | None, units: str | None) -> bool:
    """True for cost-per-outcome estimands, where a LOWER value is better.

    Recognized by kind (``cost_per_outcome``, ``cpa``) or by per-outcome money
    units (``$/conversion``). Deliberately narrow: it must not catch a channel
    whose *divisor* is dollars — every ROI has that — only a metric whose
    **value** is money spent per unit of outcome.
    """
    k = (kind or "").lower().replace(" ", "_")
    u = (units or "").lower().replace(" ", "")
    if "cost_per" in k or k in {"cpa", "cost"}:
        return True
    return u.startswith("$/") or u.startswith("cost/")


@dataclass(frozen=True)
class EvidenceReference:
    """The bar an estimand is graded against, plus how it was arrived at.

    ``value`` is ``None`` when no defensible bar exists — the state that used to
    be silently filled with ``0.0``. Callers render :attr:`hint` rather than
    reconstructing a sentence from ``value``, which is how the dashboard came to
    print "vs 0 (no effect)" beside a profit break-even of 2.5.
    """

    value: float | None
    direction: str
    hint: str
    #: "ratio" | "zero" | "declared" | "unresolved" — which rule produced it.
    basis: str

    @property
    def resolved(self) -> bool:
        return self.value is not None

    @property
    def is_ratio(self) -> bool:
        """Back-compat flag for surfaces that format ratios differently."""
        return self.basis == "ratio"


def resolve_reference(
    kind: str | None = None,
    units: str | None = None,
    *,
    explicit: float | None = None,
) -> EvidenceReference:
    """Resolve the grading bar for one estimand.

    Precedence: an ``explicit`` reference minted by the measurement layer
    (``EstimandResult.extra['metric_reference']`` — efficiency metrics carry 0
    though their kind is still "roi", and a profit basis carries ``1/margin``)
    beats every heuristic. Otherwise ratios grade against 1.0, costs are
    **unresolved**, and everything else against 0.
    """
    cost = is_cost_kind(kind, units)
    direction = LOWER_IS_BETTER if cost else HIGHER_IS_BETTER

    if explicit is not None:
        ref = float(explicit)
        if cost:
            hint = f"vs {ref:,.2f} (break-even cost)"
            basis = "declared"
        elif ref == 1.0:
            hint = "vs 1.0 (break-even)"
            basis = "ratio"
        elif ref == 0.0:
            hint = "vs 0 (no effect)"
            basis = "zero"
        else:
            # A moved bar — a profit break-even, or an efficiency reference.
            hint = f"vs {ref:g} (break-even)"
            basis = "declared"
        return EvidenceReference(value=ref, direction=direction, hint=hint, basis=basis)

    if cost:
        return EvidenceReference(
            value=None,
            direction=LOWER_IS_BETTER,
            hint="no break-even cost declared",
            basis="unresolved",
        )
    if is_ratio_kind(kind, units):
        return EvidenceReference(1.0, HIGHER_IS_BETTER, "vs 1.0 (break-even)", "ratio")
    return EvidenceReference(0.0, HIGHER_IS_BETTER, "vs 0 (no effect)", "zero")


#: Why a cost metric is ungraded, rendered wherever the verdict is "na" so the
#: blank cell states its cause instead of reading as a failed computation.
UNRESOLVED_COST_REASON = (
    "A cost per outcome is only good or bad relative to what one outcome is "
    "worth, and no value has been declared. Grading it against zero would mark "
    "every channel Strong, since a cost is above zero by construction. Set the "
    "project's KPI valuation to get a break-even cost."
)


def classify_evidence(
    *,
    status: str | None,
    mean: float | None,
    lower: float | None,
    upper: float | None,
    reference: float | EvidenceReference | None,
    direction: str = HIGHER_IS_BETTER,
) -> str:
    """Evidence label vs the reference: strong / below / uncertain / na.

    ``strong`` means the interval clears the bar in the metric's own favourable
    direction; ``below`` means it clears it the wrong way; ``uncertain`` means it
    straddles. An unresolved reference is ``na`` — the interval is fine, there
    is simply nothing to compare it to.

    ``reference`` accepts an :class:`EvidenceReference`, in which case its
    ``direction`` is used and the ``direction`` argument is ignored.
    """
    if isinstance(reference, EvidenceReference):
        direction = reference.direction
        ref_value = reference.value
    else:
        ref_value = reference

    if status not in (None, "ok") or mean is None or lower is None or upper is None:
        return "na"
    if ref_value is None:
        return "na"

    ref = float(ref_value)
    if direction == LOWER_IS_BETTER:
        if float(upper) < ref:
            return "strong"
        if float(lower) > ref:
            return "below"
        return "uncertain"

    if float(lower) > ref:
        return "strong"
    if float(upper) < ref:
        return "below"
    return "uncertain"


def verdict_label(verdict: str, direction: str = HIGHER_IS_BETTER) -> str:
    """Human label for a verdict, stated in the direction the metric runs.

    The ``below`` verdict means "credibly on the wrong side of the bar", which
    for a cost per outcome is *above* it. Rendering the token literally would
    caption a $45 CPA against a $20 break-even as "Below reference" — the
    opposite of what the interval says.
    """
    if verdict == "below":
        return (
            "Above reference" if direction == LOWER_IS_BETTER else "Below reference"
        )
    return {
        "strong": "Strong",
        "uncertain": "Uncertain",
        "na": "Not assessable",
    }.get(verdict, verdict)
