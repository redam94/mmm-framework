"""Canonical built-in estimands + capability-keyed defaults.

Each built-in is a faithful, serializable re-expression of one number the
framework already produces (the four legacy notions) plus two demonstrators that
prove the schema generalizes beyond ROI. The realization knobs
(:class:`~mmm_framework.estimands.spec.Realization`) pin the bit-stable
arithmetic — see ``technical-docs/estimands.md`` and the equivalence tests.

The four legacy notions, by built-in:

* ``contribution_roi``    ↔ ``reporting.helpers.roi.compute_roi_with_uncertainty``
  (the dashboard decomposition ROI — reads the in-graph Deterministic; the number
  the UI shows).
* ``counterfactual_roi``  ↔ ``analysis.MMMAnalyzer.compute_channel_roi``
  (zero-out predict; a *different* number).
* ``marginal_roas``       ↔ ``model.compute_marginal_contributions``.
* ``contribution``        ↔ ``model.compute_counterfactual_contributions`` totals.

Two demonstrators: ``awareness_lift`` (mean lift, no denominator) and
``cost_per_conversion`` (inverted ratio: spend / incremental conversions).
"""

from __future__ import annotations

from .capabilities import (
    HAS_CONTRIBUTION_DETERMINISTIC,
    HAS_CONTRIBUTIONS,
    HAS_LATENT,
)
from .spec import (
    ALL_CHANNELS,
    Contrast,
    Contribution,
    Estimand,
    LatentVar,
    MarginalSpend,
    Observed,
    ObservedInput,
    Outcome,
    Realization,
    Summary,
    ZeroInput,
)

#: Default +10% perturbation for the marginal estimand (matches
#: ``compute_marginal_contributions``' default ``spend_increase_pct=10``).
DEFAULT_MARGINAL_FACTOR = 1.1

_PROB_SUMMARIES = [
    Summary(name="prob_positive", threshold=0.0, side="gt"),
    Summary(name="prob_profitable", threshold=1.0, side="gt"),
]


def _contribution_roi() -> Estimand:
    """Dashboard decomposition ROI (the UI number)."""
    return Estimand(
        name="contribution_roi",
        kind="roi",
        numerator=Contribution(target=ALL_CHANNELS, source="in_graph_deterministic"),
        denominator=ObservedInput(target=ALL_CHANNELS, source="panel"),
        op_ratio_zero_denominator="skip",
        realization=Realization(point_rule="mean_of_samples", hdi_method="az_hdi"),
        summaries=list(_PROB_SUMMARIES),
        required_capabilities=[HAS_CONTRIBUTION_DETERMINISTIC],
        units="ROI",
        causal_assumptions=(
            "Decomposition ROI from the in-graph channel_contributions "
            "Deterministic over the full period; assumes the fitted additive "
            "decomposition is causal."
        ),
    )


def _counterfactual_roi() -> Estimand:
    """Zero-out counterfactual ROI (differs from the decomposition ROI)."""
    return Estimand(
        name="counterfactual_roi",
        kind="roi",
        numerator=Contrast(
            quantity=Outcome(),
            intervention=Observed(),
            baseline=ZeroInput(target=ALL_CHANNELS),
            op="difference",
            reduce="sum",
            paired_seed=False,
        ),
        denominator=ObservedInput(target=ALL_CHANNELS, source="raw"),
        op_ratio_zero_denominator="zero",
        realization=Realization(point_rule="diff_of_means", hdi_method="percentile"),
        required_capabilities=[HAS_CONTRIBUTIONS],
        units="ROI",
        causal_assumptions=(
            "Incremental KPI from turning the channel off (predict(observed) − "
            "predict(channel→0)); unpaired posterior-predictive draws."
        ),
    )


def _marginal_roas() -> Estimand:
    """Marginal ROAS from a +10% spend perturbation."""
    return Estimand(
        name="marginal_roas",
        kind="marginal_roas",
        numerator=Contrast(
            quantity=Outcome(),
            intervention={
                "type": "scale_input",
                "target": ALL_CHANNELS,
                "factor": DEFAULT_MARGINAL_FACTOR,
            },
            baseline=Observed(),
            op="difference",
            reduce="sum",
            paired_seed=True,
        ),
        denominator=MarginalSpend(target=ALL_CHANNELS, intervention_ref="numerator"),
        op_ratio_zero_denominator="zero",
        realization=Realization(
            point_rule="diff_of_means", hdi_method="finite_percentile"
        ),
        required_capabilities=[HAS_CONTRIBUTIONS],
        units="mROAS",
        causal_assumptions=(
            "Incremental KPI per incremental dollar from a small (+10%) spend "
            "perturbation; paired posterior-predictive draws so observation noise "
            "cancels."
        ),
    )


def _contribution() -> Estimand:
    """Total incremental KPI contribution of the channel (no denominator)."""
    return Estimand(
        name="contribution",
        kind="contribution",
        numerator=Contrast(
            quantity=Outcome(),
            intervention=Observed(),
            baseline=ZeroInput(target=ALL_CHANNELS),
            op="difference",
            reduce="sum",
            paired_seed=False,
        ),
        denominator=None,
        realization=Realization(point_rule="diff_of_means", hdi_method="percentile"),
        required_capabilities=[HAS_CONTRIBUTIONS],
        units="KPI",
        causal_assumptions="Incremental KPI from turning the channel off.",
    )


def _awareness_lift() -> Estimand:
    """Mean per-period lift of the outcome attributable to the channel.

    A demonstrator for ``reduce=mean`` with no denominator. Targeting a latent
    awareness state instead of the outcome is a one-line change (swap the
    ``Outcome`` quantity for ``LatentVar(name=...)`` and require
    ``HAS_LATENT:<name>``).
    """
    return Estimand(
        name="awareness_lift",
        kind="lift",
        numerator=Contrast(
            quantity=Outcome(),
            intervention=Observed(),
            baseline=ZeroInput(target=ALL_CHANNELS),
            op="difference",
            reduce="mean",
            paired_seed=False,
        ),
        denominator=None,
        realization=Realization(point_rule="diff_of_means", hdi_method="percentile"),
        required_capabilities=[HAS_CONTRIBUTIONS],
        units="KPI/period",
        causal_assumptions="Mean per-period incremental outcome from the channel.",
    )


def _cost_per_conversion() -> Estimand:
    """Spend per incremental conversion (an inverted ratio demonstrator)."""
    return Estimand(
        name="cost_per_conversion",
        kind="cost_per_outcome",
        numerator=ObservedInput(target=ALL_CHANNELS, source="raw"),
        denominator=Contrast(
            quantity=Outcome(),
            intervention=Observed(),
            baseline=ZeroInput(target=ALL_CHANNELS),
            op="difference",
            reduce="sum",
            paired_seed=False,
        ),
        op_ratio_zero_denominator="nan",
        realization=Realization(point_rule="diff_of_means", hdi_method="percentile"),
        required_capabilities=[HAS_CONTRIBUTIONS],
        units="$/conversion",
        causal_assumptions="Observed spend divided by incremental conversions.",
    )


# =============================================================================
# Latent-scalar estimands — for non-MMM families (CFA/LCA/…)
# =============================================================================
# A bare ``LatentVar`` quantity summarizing a per-draw scalar posterior variable
# (a fit index, a class size, a named scalar loading) as mean + HDI. Gated by the
# auto-exposed ``HAS_LATENT:<var>`` capability, so it cleanly returns
# ``unsupported`` on a model that doesn't carry that variable.


def latent_scalar(
    name: str,
    *,
    var: str | None = None,
    kind: str = "latent",
    units: str = "",
    hdi_prob: float = 0.94,
    causal_assumptions: str = "Posterior summary of a fitted latent quantity.",
) -> Estimand:
    """A declarative estimand that summarizes the scalar posterior variable
    ``var`` (default: ``name``). Used by non-MMM families to surface fit indices,
    factor loadings, class sizes, etc. through the estimand engine."""
    v = var or name
    return Estimand(
        name=name,
        kind=kind,
        numerator=LatentVar(name=v),
        denominator=None,
        realization=Realization(point_rule="mean_of_samples", hdi_method="percentile"),
        required_capabilities=[f"{HAS_LATENT}:{v}"],
        units=units,
        causal_assumptions=causal_assumptions,
    )


def fit_index(name: str, *, var: str | None = None) -> Estimand:
    """A model fit index (e.g. ``cfi``, ``tli``, ``srmr``) read per draw."""
    return latent_scalar(
        name,
        var=var,
        kind="fit_index",
        units="index",
        causal_assumptions=(
            "Per-draw model fit index from the fitted posterior (model-implied vs "
            "observed moments)."
        ),
    )


def factor_loading(name: str, *, var: str | None = None) -> Estimand:
    """A single (named, scalar) factor loading read per draw."""
    return latent_scalar(
        name,
        var=var,
        kind="factor_loading",
        units="loading",
        causal_assumptions="Posterior of a fitted factor loading.",
    )


#: name -> zero-arg factory returning a fresh ``Estimand`` (avoids shared
#: mutable state across callers).
BUILTINS = {
    "contribution_roi": _contribution_roi,
    "counterfactual_roi": _counterfactual_roi,
    "marginal_roas": _marginal_roas,
    "contribution": _contribution,
    "awareness_lift": _awareness_lift,
    "cost_per_conversion": _cost_per_conversion,
}

#: The MMM default list surfaced when a model declares no estimands of its own.
DEFAULT_NAMES = ["contribution_roi", "marginal_roas", "contribution"]


def get(name: str) -> Estimand:
    """A fresh copy of the named built-in estimand. Raises ``KeyError`` if unknown."""
    return BUILTINS[name]()


def all_builtins() -> list[Estimand]:
    """Fresh copies of every built-in estimand."""
    return [factory() for factory in BUILTINS.values()]


def defaults_for(capabilities: set[str]) -> list[Estimand]:
    """Default estimands whose ``required_capabilities`` are met by ``capabilities``.

    Keyed by **capability, not class name**, so a non-MMM model that lacks (say)
    ``HAS_CONTRIBUTION_DETERMINISTIC`` auto-filters ``contribution_roi`` rather
    than erroring.
    """
    out = []
    for name in DEFAULT_NAMES:
        est = get(name)
        if all(c in capabilities for c in est.required_capabilities):
            out.append(est)
    return out


__all__ = [
    "BUILTINS",
    "DEFAULT_NAMES",
    "DEFAULT_MARGINAL_FACTOR",
    "get",
    "all_builtins",
    "defaults_for",
    "latent_scalar",
    "fit_index",
    "factor_loading",
]
