"""Which estimation paradigm produced a number, and what may be said about it.

Every report section, estimand label and diagnostic in this codebase was written
assuming a posterior. Epic #180 adds an estimator that does not produce one, so
this module is the single place that answers two questions:

* **Which family is this fit?** :func:`family_of` reads ``inference_family`` out
  of a diagnostics dict. **Absence reads as Bayesian** — every fit that predates
  the frequentist path, and every trace loaded from an older save, has no such
  key and is one.
* **What is the interval called?** :func:`interval_noun` and
  :func:`interval_phrase`. A bootstrap percentile interval is a **confidence**
  interval: it describes the sampling variability of an estimator, not a
  probability distribution over a parameter. "There is a 90% probability the ROI
  is in this range" is true of a credible interval and false of this one, and
  that sentence is currently written into a dozen report surfaces.

Why this lives in ``diagnostics`` rather than ``reporting``: the same
distinction gates convergence verdicts (:mod:`~mmm_framework.diagnostics.convergence`),
serializer metadata, and ``planning/history`` run metrics, none of which import
the reporting stack. The vocabulary and the gate have to agree, so they share a
module.

The provenance contract, stamped by every frequentist fit into
``results.diagnostics`` and the ``InferenceData`` attrs, is specified in
``technical-docs/frequentist-estimation.md`` §8.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BAYESIAN",
    "FREQUENTIST",
    "estimator_label",
    "family_of",
    "frequentist_caveats",
    "interval_noun",
    "interval_phrase",
    "is_frequentist",
    "not_applicable_reason",
]

#: The two paradigms. Stored as plain strings so they round-trip through JSON,
#: ``configs.json`` and the ``InferenceData`` attrs without an enum import.
BAYESIAN = "bayesian"
FREQUENTIST = "frequentist"

_ESTIMATOR_LABELS = {
    "ridge": "penalized ridge regression",
    "constrained": "constrained least squares (convex program)",
}

#: Plain-language reason each posterior-only surface is unavailable, keyed by a
#: short slug the caller passes. Kept here so the report, the deck and the
#: interactive page give the same explanation rather than three paraphrases.
_NOT_APPLICABLE = {
    "convergence": (
        "R-hat, effective sample size and divergences describe an MCMC "
        "sampler. This fit is a penalized point estimate with bootstrap "
        "replicates — there is no chain to assess, so convergence is not "
        "applicable rather than passing."
    ),
    "posterior_predictive": (
        "A posterior-predictive check compares replicated datasets drawn from "
        "the posterior against the observed one, and a Bayesian p-value is a "
        "tail probability under that distribution. A frequentist fit has no "
        "posterior to draw from, so both are undefined here."
    ),
    "prior": (
        "This fit has no prior. The transforms were selected by out-of-sample "
        "predictive error and the coefficients by a penalized least-squares "
        "solve, so prior-predictive checks, prior-vs-posterior contraction and "
        "simulation-based calibration have no analogue."
    ),
    "learning": (
        "Prior-to-posterior contraction measures how much the data moved a "
        "belief. A frequentist fit starts from no belief, so there is nothing "
        "to contract."
    ),
}


def family_of(diagnostics: Any) -> str:
    """The inference family a diagnostics dict describes.

    Absence of the key reads as :data:`BAYESIAN`, which is what makes this safe
    to call on any fit produced before the frequentist path existed.
    """
    if not isinstance(diagnostics, dict):
        return BAYESIAN
    value = diagnostics.get("inference_family")
    return FREQUENTIST if str(value).lower() == FREQUENTIST else BAYESIAN


def is_frequentist(diagnostics: Any) -> bool:
    """``True`` when this fit came from the frequentist path."""
    return family_of(diagnostics) == FREQUENTIST


def interval_noun(family: str | Any = BAYESIAN, *, plural: bool = False) -> str:
    """``"credible interval"`` or ``"confidence interval"``.

    Accepts either a family string or a diagnostics dict, because call sites
    have one or the other and converting at every one of them is how a surface
    gets missed.
    """
    fam = family if isinstance(family, str) else family_of(family)
    noun = "confidence interval" if fam == FREQUENTIST else "credible interval"
    return noun + "s" if plural else noun


def interval_phrase(ci_pct: float | int, family: str | Any = BAYESIAN) -> str:
    """e.g. ``"90% credible interval"`` / ``"90% bootstrap confidence interval"``."""
    fam = family if isinstance(family, str) else family_of(family)
    pct = int(round(float(ci_pct) * 100)) if float(ci_pct) <= 1 else int(ci_pct)
    if fam == FREQUENTIST:
        return f"{pct}% bootstrap confidence interval"
    return f"{pct}% credible interval"


def estimator_label(diagnostics: Any) -> str:
    """Human-readable estimator name, e.g. ``"penalized ridge regression"``."""
    if not isinstance(diagnostics, dict):
        return "estimator"
    key = str(diagnostics.get("estimator") or "").lower()
    return _ESTIMATOR_LABELS.get(key, key or "estimator")


def not_applicable_reason(slug: str) -> str:
    """Why a posterior-only surface is unavailable for a frequentist fit.

    Gating with an explanation rather than a blank space is the whole point: a
    missing convergence table reads as an oversight, a stated "not applicable
    because there is no chain" reads as a property of the method.
    """
    return _NOT_APPLICABLE.get(
        slug,
        "This view requires a posterior, which a frequentist fit does not produce.",
    )


def frequentist_caveats(diagnostics: Any) -> list[str]:
    """The statements that must ride with any rendered frequentist number.

    Prefers the caveats the estimator itself recorded (they carry the fit's own
    effective degrees of freedom and near-optimal-candidate count); falls back to
    a generic set so a reloaded fit whose diagnostics were trimmed still says
    something true.
    """
    if isinstance(diagnostics, dict):
        stored = diagnostics.get("caveats")
        if isinstance(stored, list) and stored:
            return [str(c) for c in stored]
    return [
        "These are bootstrap confidence intervals from a frequentist point "
        "estimate, not credible intervals: they describe the sampling "
        "variability of the estimator, not a probability distribution over the "
        "parameter.",
        "Ridge is biased by construction, so an interval covers the estimator's "
        "sampling distribution rather than the true parameter.",
    ]
