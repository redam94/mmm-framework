"""The one registry of what ``inference.method`` accepts, and what each means.

``spec["inference"]["method"]`` takes a **union** vocabulary: the seven
:class:`~mmm_framework.config.enums.FitMethod` members plus the two frequentist
:class:`~mmm_framework.config.enums.InferenceMethod` members, because that one
field selects a Bayesian estimator *or* leaves the Bayesian paradigm entirely
(``agents/fitting.py`` branches on exactly that). Nothing owned that union, so
it had been copied out by hand four times — a validation set in
``agents/fitting.py``, a label list and an ``EXACT_METHODS`` set in
``ModelSpecWidget.tsx``, and an inline three-branch test in
``ArtifactsPanel.tsx``.

The copies drifted, and the drift was user-visible: the two frontend mirrors
were written before the frequentist path shipped, so both spelled
``is_approximate`` as ``method not in (nuts, smc)``. A v1.3
``inference.method = 'frequentist_ridge'`` spec therefore rendered an amber
**"approximate"** badge and the sentence "re-fit with NUTS before trusting
intervals", flatly contradicting the shipped rule that ``approximate`` stays
**False** for a frequentist fit (#188). A penalized point estimate with
bootstrap confidence intervals is not an approximation of a posterior — it is a
different quantity, and the honest caveat is a different sentence.

So the registry carries the caveat too, not just the flag. The frontend reads a
**generated** mirror of this module (``frontend/src/api/generated/inferenceMethods.ts``,
emitted by ``scripts/gen_fe_enums.py``) and ``tests/test_fe_enum_mirror.py``
fails when the two diverge — the same shape as the ``tests/contracts/`` REST
snapshot gate. A hand-copied enum in the client is fine; a hand-copied enum with
nothing checking it is how this bug shipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import FitMethod, InferenceMethod

__all__ = [
    "BAYESIAN",
    "FREQUENTIST",
    "INFERENCE_METHODS",
    "InferenceMethodInfo",
    "frequentist_method_values",
    "method_info",
    "method_values",
]

#: Paradigm names. Deliberately the same strings as
#: ``diagnostics.provenance.BAYESIAN`` / ``FREQUENTIST`` — a test pins the
#: equality rather than importing across the layer.
BAYESIAN = "bayesian"
FREQUENTIST = "frequentist"

_APPROXIMATE_CAVEAT = (
    "Approximate fits run in seconds for model checking, but their uncertainty "
    "is not calibrated — re-fit with NUTS before trusting intervals or making "
    "spend decisions."
)
_SMC_CAVEAT = (
    "SMC is an exact sampler for multimodal posteriors and yields a log "
    "marginal likelihood for model comparison. It is not a speedup."
)
_FREQUENTIST_CAVEAT = (
    "A penalized point estimate with bootstrap CONFIDENCE intervals — not a "
    "posterior. Convergence diagnostics, posterior-predictive checks and "
    "prior-based views do not apply; they are reported as not applicable "
    "rather than passing."
)


@dataclass(frozen=True)
class InferenceMethodInfo:
    """What one ``inference.method`` value selects, and how to describe it."""

    value: str
    label: str
    paradigm: str
    #: ``FitMethod.is_approximate`` for the Bayesian estimators. **False** for
    #: the frequentist ones: they are not approximations of a posterior.
    approximate: bool
    #: What the reported interval is called (``diagnostics.provenance``).
    interval_kind: str
    #: One sentence a surface renders beside the method; ``None`` when the
    #: method needs no disclosure (NUTS).
    caveat: str | None = None

    @property
    def is_frequentist(self) -> bool:
        return self.paradigm == FREQUENTIST


def _bayesian(method: FitMethod, label: str, caveat: str | None) -> InferenceMethodInfo:
    return InferenceMethodInfo(
        value=method.value,
        label=label,
        paradigm=BAYESIAN,
        # Read off the enum rather than restated, so the "SMC is exact" rule has
        # exactly one home.
        approximate=method.is_approximate,
        interval_kind="credible",
        caveat=caveat,
    )


def _frequentist(method: InferenceMethod, label: str) -> InferenceMethodInfo:
    return InferenceMethodInfo(
        value=method.value,
        label=label,
        paradigm=FREQUENTIST,
        approximate=False,
        interval_kind="confidence",
        caveat=_FREQUENTIST_CAVEAT,
    )


#: Every accepted value, in the order a picker should offer them: exact Bayesian
#: samplers, then approximate Bayesian fits, then the frequentist estimators.
INFERENCE_METHODS: tuple[InferenceMethodInfo, ...] = (
    _bayesian(FitMethod.NUTS, "NUTS (full MCMC)", None),
    _bayesian(FitMethod.SMC, "SMC (Sequential Monte Carlo)", _SMC_CAVEAT),
    _bayesian(FitMethod.MAP, "MAP (point estimate)", _APPROXIMATE_CAVEAT),
    _bayesian(FitMethod.LAPLACE, "Laplace (MAP + Gaussian)", _APPROXIMATE_CAVEAT),
    _bayesian(FitMethod.ADVI, "ADVI (variational)", _APPROXIMATE_CAVEAT),
    _bayesian(
        FitMethod.FULLRANK_ADVI, "Full-rank ADVI (variational)", _APPROXIMATE_CAVEAT
    ),
    _bayesian(FitMethod.PATHFINDER, "Pathfinder", _APPROXIMATE_CAVEAT),
    _frequentist(InferenceMethod.FREQUENTIST_RIDGE, "Ridge (penalized, bootstrap CIs)"),
    _frequentist(
        InferenceMethod.FREQUENTIST_CVXPY, "Constrained LS (convex, bootstrap CIs)"
    ),
)

_BY_VALUE = {m.value: m for m in INFERENCE_METHODS}


def method_info(value: str | None) -> InferenceMethodInfo | None:
    """Look up a method descriptor; ``None`` for an unrecognized value.

    Returns ``None`` rather than guessing, because the guess is the bug: the
    frontend's ``!(nuts|smc)`` fallback classified every value it had never
    heard of as an approximate Bayesian fit.
    """
    if value is None:
        return None
    return _BY_VALUE.get(str(value).strip().lower())


def method_values() -> set[str]:
    """Every accepted ``inference.method`` value."""
    return set(_BY_VALUE)


def frequentist_method_values() -> set[str]:
    """The subset that leaves the Bayesian paradigm entirely."""
    return {m.value for m in INFERENCE_METHODS if m.is_frequentist}
