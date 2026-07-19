"""Experiment-method registry — named methodologies (design + estimate + power +
data-requirement) for the measurement loop.

Importing this package registers the built-in methods and wires the new geo
estimators into :data:`planning.simulation._GEO_ESTIMATORS` so the A/A·A/B
methodology leaderboard runs them automatically. Enumerate methods via
:func:`list_methods` / :func:`methods_for_data`; fetch one via :func:`get_method`.

Phase 2 (``ghost_ads``, ``switchback``) registers into the same table.
"""

from __future__ import annotations

from .base import (
    DataRequirement,
    MethodSpec,
    get_method,
    has_method,
    list_methods,
    methods_for_data,
    register,
)
from .did_mmt import did_mmt_estimator
from .gbr import gbr_estimator, gbr_iroas
from .ghost_ads import (
    GhostAdsDesign,
    ghost_ads_power,
    ghost_ads_power_at,
    ghost_ads_simulate,
    ghost_ads_users_for_mde,
)
from .switchback import switchback_design, switchback_power
from .synthetic_control import (
    synthetic_control_analysis,
    synthetic_control_estimator,
    synthetic_control_weights,
)
from .tbr import tbr_causal_impact, tbr_counterfactual, tbr_estimator

_GEO = "geo"

# ── Built-in geo methods ──────────────────────────────────────────────────────

register(
    MethodSpec(
        key="synthetic_control",
        name="Synthetic control",
        requirement=DataRequirement(
            family=_GEO,
            min_geos=4,
            needs_panel=True,
            min_pre_weeks=12,
            notes="Convex donor weights + placebo-permutation inference.",
        ),
        estimator_fn=synthetic_control_estimator,
        references=("docs/blog-synthetic-control.html",),
        description=(
            "Weights a donor pool of control geos (convex: no extrapolation) to "
            "reconstruct the treated market's counterfactual; inference by placebo "
            "permutation over donors."
        ),
    )
)

register(
    MethodSpec(
        key="regadj_geo",
        name="Regression-adjusted geo (ridge)",
        requirement=DataRequirement(
            family=_GEO,
            min_geos=4,
            needs_panel=True,
            min_pre_weeks=12,
            notes="Unconstrained ridge control weights (can extrapolate).",
        ),
        estimator_fn=None,  # lives in simulation.regadj_geo_estimator (already registered)
        references=(),
        description=(
            "Ridge-weighted control regression scored on the test window. Faster "
            "and lower-variance than SCM but can extrapolate outside the donor hull "
            "— compare against synthetic control."
        ),
    )
)

register(
    MethodSpec(
        key="tbr",
        name="Time-based regression (TBR / CausalImpact)",
        requirement=DataRequirement(
            family=_GEO,
            min_geos=2,
            needs_panel=True,
            min_pre_weeks=12,
            notes="Bayesian structural time-series counterfactual.",
        ),
        estimator_fn=tbr_estimator,
        references=("docs/blog-geo-experiments-tbr.html",),
        description=(
            "Regresses the treated time series on the control series over the "
            "pre-period and projects the counterfactual through the test window; "
            "cumulative effect with a full posterior (BSTS for the headline)."
        ),
    )
)

register(
    MethodSpec(
        key="gbr",
        name="Geo-based regression (GBR)",
        requirement=DataRequirement(
            family=_GEO,
            min_geos=4,
            needs_panel=True,
            min_pre_weeks=8,
            notes="Cross-sectional post-on-pre geo regression.",
        ),
        estimator_fn=gbr_estimator,
        references=(),
        description=(
            "One row per geo: regress test-period response on pre-period response "
            "and treatment; the treatment coefficient is the incremental effect "
            "(iROAS when divided by treated spend)."
        ),
    )
)

register(
    MethodSpec(
        key="did_mmt",
        name="Matched-market DiD (DiD-MMT)",
        requirement=DataRequirement(
            family=_GEO,
            min_geos=4,
            needs_panel=True,
            min_pre_weeks=8,
            notes="Matched treatment/control markets, cluster-robust per-pair DiD.",
        ),
        estimator_fn=did_mmt_estimator,
        references=("docs/blog-staggered-did.html",),
        description=(
            "Difference-in-differences on matched test/control markets "
            "(observational when not randomized); cluster-robust per-pair SE."
        ),
    )
)


# ── Non-geo methods (Phase 2): ghost ads + switchback ─────────────────────────

register(
    MethodSpec(
        key="ghost_ads",
        name="Ghost ads (user-level RCT)",
        requirement=DataRequirement(
            family="user",
            needs_panel=False,
            needs_pre_period=False,
            notes="Individual randomization via the ad server; needs user-level "
            "reach + conversion counts, not a geo panel.",
        ),
        estimator_fn=None,  # user-level analysis; ghost_ads_simulate covers power
        power_fn=ghost_ads_power,
        references=(),
        description=(
            "User-level incrementality: treated users see the real ad, ghost/PSA "
            "users a placebo. Two-proportion (or count/revenue) lift with ITT vs "
            "treatment-on-treated dilution — a standalone pre-fit power calculator."
        ),
    )
)

register(
    MethodSpec(
        key="switchback",
        name="Switchback (time-randomized)",
        requirement=DataRequirement(
            family="switchback",
            needs_panel=False,
            min_pre_weeks=8,
            notes="National time series; block length must exceed carryover memory.",
        ),
        design_fn=switchback_design,
        estimator_fn=None,  # analysis = simulation.national_onoff_estimator
        references=("docs/blog-switchback-experiments.html",),
        description=(
            "Randomized on/off block schedule on the national series; power "
            "honest to carryover (block ≥ adstock memory + burn-in) and "
            "autocorrelation (AR(1) design effect)."
        ),
    )
)


def _wire_simulation_estimators() -> None:
    """Add the new geo estimators to the leaderboard's ``_GEO_ESTIMATORS`` so
    A/A·A/B runs them. They recover the full treated-total effect → the default
    ``_estimand_scale`` of 1.0 is correct (unlike the per-pair DiD). Switchback's
    analysis is the on/off contrast, registered under its own key."""
    from .. import simulation

    simulation._GEO_ESTIMATORS.setdefault(
        "synthetic_control", synthetic_control_estimator
    )
    simulation._GEO_ESTIMATORS.setdefault("tbr", tbr_estimator)
    simulation._GEO_ESTIMATORS.setdefault("gbr", gbr_estimator)
    simulation._NATIONAL_ESTIMATORS.setdefault(
        "switchback", simulation.national_onoff_estimator
    )


_wire_simulation_estimators()


__all__ = [
    "DataRequirement",
    "MethodSpec",
    "register",
    "get_method",
    "has_method",
    "list_methods",
    "methods_for_data",
    "synthetic_control_estimator",
    "synthetic_control_analysis",
    "synthetic_control_weights",
    "tbr_estimator",
    "tbr_counterfactual",
    "tbr_causal_impact",
    "gbr_estimator",
    "gbr_iroas",
    "did_mmt_estimator",
    "GhostAdsDesign",
    "ghost_ads_power",
    "ghost_ads_power_at",
    "ghost_ads_users_for_mde",
    "ghost_ads_simulate",
    "switchback_design",
    "switchback_power",
]
