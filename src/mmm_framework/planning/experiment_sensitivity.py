"""How much residual bias could an experiment readout be carrying?

A geo lift test is not automatically clean. What threatens it depends entirely on
how it was assigned, and conflating the cases is the mistake this module exists
to prevent:

* A **matched-market DiD** is observational. Nothing was randomized; the estimate
  rests on parallel trends, and unmeasured confounding is exactly the threat.
* A **randomized** geo split has no confounding problem at all — assignment was
  exogenous by construction. What remains is interference (a national buy
  bleeding across DMA lines, consumers crossing borders, a platform reallocating
  budget out of the holdout) and concurrent shocks hitting one arm. Spillover
  *attenuates*, so the honest correction is toward a larger effect, not a smaller
  one.
* A **user-level RCT** (ghost ads) is cleaner still: the residual issues are
  measurement (ghost-log fidelity, cross-device identity loss, ITT vs TOT) and
  external validity — the platform's lift among its addressable users is not the
  channel's marginal ROI in the MMM's aggregate panel.

Handing a randomized design a "confounding tipping point" would dress up the one
threat it does not face. So every report here names the threat it is pricing.

The prior is measured, not asserted
-----------------------------------
The number that matters is the width of the bias prior, and a design already
produces evidence for it. ``planning.design._placebo_did`` scores every
historical window as though it were the experiment: the spread of those placebo
"lifts" is what the estimator returns when nothing happened. That spread contains
sampling noise *and* whatever differential drift the pair structure carries, and
the analytic DiD standard error models only the first — so the **excess** is the
defensible bias prior::

    bias_sigma = sqrt(max(placebo_sd**2 - analytic_sd**2, 0))

**The double-counting trap.** ``geo_lift_design`` already multiplies its own
standard error by ``calibration_factor = placebo_sd / analytic_sd`` whenever it
has at least twelve placebo windows (``se_source == "placebo_calibrated"``). If
the readout's standard error was taken from that design, the inflation is already
inside it and adding the excess again would inflate every calibrated interval by
a spurious factor. :func:`derive_bias_prior` therefore compares the readout's
standard error against the design's and reports which regime it is in
(:attr:`DerivedBiasPrior.absorbed`) rather than guessing.

A floor for what no placebo can see
-----------------------------------
Placebo windows cannot see spillover, since no treatment ever ran in them. Each
method therefore carries a floor — a relative bias width standing for the threats
its design leaves open — combined in quadrature with the measured excess. Floors
are **assumptions**; they are labelled as such in ``bias_source`` so a report can
never present one as a measurement.

Why design-derived widths are ROAS-only
---------------------------------------
A design's evidence lives on its own scale, and moving it to another estimand is
where the silent errors are: the placebo runs on the *average matched pair* while
most estimators return the *whole treated cell*; ``weekly_spend_delta`` is stored
unsigned even though a holdout's delta is negative; and the planned spend delta
is not the realized one. Since ``geo_lift_design`` already reports ``se_roas`` —
the design's own standard error on the ROAS scale — deriving a ROAS-scale bias
needs *no conversion at all*.

So automatic derivation is offered for ``roas`` and refused for ``contribution``
and ``mroas``, which need an explicitly supplied number. The floor, being
relative to the measured value, still applies to every estimand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..diagnostics.bias_sensitivity import (
    DEFAULT_DECISION_THRESHOLD,
    BiasPrior,
    BiasSensitivity,
    bias_sensitivity_report,
)

__all__ = [
    "DerivedBiasPrior",
    "DesignThreat",
    "ExperimentSensitivityReport",
    "METHOD_THREATS",
    "UNKNOWN_METHOD_THREAT",
    "derive_bias_prior",
    "experiment_sensitivity",
    "threat_for",
]


@dataclass(frozen=True)
class DesignThreat:
    """What a design actually leaves open, and how wide to assume it is."""

    method_key: str
    label: str
    #: ``"unmeasured_confounding"`` | ``"interference"`` | ``"external_validity"``
    threat: str
    #: Relative bias width standing in for the threats no placebo window can see.
    floor: float
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_key": self.method_key,
            "label": self.label,
            "threat": self.threat,
            "floor": float(self.floor),
            "caveat": self.caveat,
        }


_RANDOMIZED_GEO = DesignThreat(
    method_key="geo_lift_randomized",
    label="Randomized geo lift",
    threat="interference",
    floor=0.10,
    caveat=(
        "Randomization makes assignment exogenous, so unmeasured confounding is "
        "NOT the threat here. What remains is interference — a national buy "
        "bleeding across market lines, consumers crossing borders, a platform "
        "reallocating budget out of the holdout — and a concurrent shock landing "
        "on one arm. Spillover attenuates a lift test, so the true effect is more "
        "likely above the measured value than below it."
    ),
)

METHOD_THREATS: dict[str, DesignThreat] = {
    "did_mmt": DesignThreat(
        method_key="did_mmt",
        label="Matched-market DiD",
        threat="unmeasured_confounding",
        floor=0.25,
        caveat=(
            "Assignment was not randomized, so parallel trends is an assumption "
            "rather than a design guarantee. The placebo band — how much 'lift' "
            "the pre-period manufactures by chance — is the direct evidence for "
            "the width below."
        ),
    ),
    "regadj_geo": DesignThreat(
        method_key="regadj_geo",
        label="Regression-adjusted geo",
        threat="unmeasured_confounding",
        floor=0.15,
        caveat=(
            "The adjustment conditions on the pre-period relationship. Anything "
            "that moved between pre and post and correlates with assignment lands "
            "in the treatment coefficient."
        ),
    ),
    "synthetic_control": DesignThreat(
        method_key="synthetic_control",
        label="Synthetic control",
        threat="unmeasured_confounding",
        floor=0.20,
        caveat=(
            "Simplex weights and donor placebos bound extrapolation, not "
            "confounding. Validity rests on no donor having been treated or "
            "shocked during the window, and on the pre-period fit being good "
            "enough for the weights to transfer."
        ),
    ),
    "tbr": DesignThreat(
        method_key="tbr",
        label="Time-based regression",
        threat="unmeasured_confounding",
        floor=0.20,
        caveat=(
            "The counterfactual projects the pre-period control-to-treated "
            "relationship through the test window. Its interval prices sampling "
            "noise, not a regime change in that relationship during the test."
        ),
    ),
    "gbr": DesignThreat(
        method_key="gbr",
        label="Geo-based regression",
        threat="unmeasured_confounding",
        floor=0.15,
        caveat=(
            "GBR conditions on the pre-period response only. A geo-level covariate "
            "that moved between pre and post and correlates with assignment is "
            "absorbed into the treatment coefficient."
        ),
    ),
    "switchback": DesignThreat(
        method_key="switchback",
        label="Switchback",
        threat="interference",
        floor=0.15,
        caveat=(
            "Time randomization removes cross-sectional confounding but not "
            "carryover leaking across block boundaries, and not a national shock "
            "aligned with the block cycle. The AR(1) design effect prices "
            "autocorrelation, not carryover smear."
        ),
    ),
    "national_flighting": DesignThreat(
        method_key="national_flighting",
        label="National flighting",
        threat="interference",
        floor=0.20,
        caveat=(
            "Budget-neutral randomized levels manufacture exogenous spend "
            "variation. The residual threat is the schedule's cycle aligning with "
            "an unmodelled seasonal or competitive one, and incomplete washout "
            "between blocks."
        ),
    ),
    "ghost_ads": DesignThreat(
        method_key="ghost_ads",
        label="Ghost ads (user-level RCT)",
        threat="external_validity",
        floor=0.10,
        caveat=(
            "Individual randomization at the ad server is the cleanest "
            "identification available: there is no unmeasured confounder. What "
            "remains is measurement (ghost-log fidelity, cross-device identity "
            "loss, ITT versus TOT dilution) and an external-validity gap — the "
            "platform's lift among its addressable users is not the channel's "
            "marginal ROI in an aggregate panel."
        ),
    ),
    "geo_lift_randomized": _RANDOMIZED_GEO,
}

#: Used when the method is unknown or unmapped. Deliberately the widest floor:
#: not knowing how an experiment was run is itself a reason for caution.
UNKNOWN_METHOD_THREAT = DesignThreat(
    method_key="unknown",
    label="Unspecified design",
    threat="unmeasured_confounding",
    floor=0.25,
    caveat=(
        "The design was not identified, so no design-specific evidence could be "
        "used and the widest default applies. Record the method to get a bias "
        "prior derived from this design's own placebo distribution instead of a "
        "default."
    ),
)

#: Extra width added when the geo design's own parallel-trends screen is failing.
_PARALLEL_TRENDS_PENALTY = 0.15
#: Extra width when a switchback's blocks are shorter than the adstock washout.
_CARRYOVER_PENALTY = 0.15


def threat_for(
    method_key: str | None, design: dict[str, Any] | None = None
) -> DesignThreat:
    """Resolve the threat model for a method, honouring the design's own flags.

    A geo design that reports ``randomized=True`` is re-pointed at the
    interference threat even when its estimator key is a DiD one, because what
    matters for this question is how units were assigned, not which estimator was
    run afterwards.
    """
    design = design or {}
    base: DesignThreat | None = None
    if method_key:
        base = METHOD_THREATS.get(str(method_key))
    if base is None:
        key = str(design.get("design_key") or "")
        if key in ("geo_lift", "matched_market_did"):
            base = METHOD_THREATS["did_mmt"]
        elif key == "national_flighting":
            base = METHOD_THREATS["national_flighting"]
    if base is None:
        base = UNKNOWN_METHOD_THREAT

    if design.get("randomized") and base.threat == "unmeasured_confounding":
        base = _RANDOMIZED_GEO

    floor = base.floor
    caveat = base.caveat
    diagnostics = design.get("diagnostics") or {}
    if diagnostics.get("parallel_trends_warning"):
        floor += _PARALLEL_TRENDS_PENALTY
        caveat += (
            " The design's own parallel-trends screen is failing (matched pairs "
            "co-move weakly), which widens the assumed bias further."
        )
    if design.get("carryover_warning"):
        floor += _CARRYOVER_PENALTY
        caveat += (
            " The design's blocks are shorter than the channel's adstock washout, "
            "so treatment leaks across block boundaries."
        )
    if floor == base.floor and caveat == base.caveat:
        return base
    return DesignThreat(
        method_key=base.method_key,
        label=base.label,
        threat=base.threat,
        floor=floor,
        caveat=caveat,
    )


# --------------------------------------------------------------------------- #
# deriving the prior
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DerivedBiasPrior:
    """A bias prior for one readout, with every component accounted for."""

    prior: BiasPrior
    threat: DesignThreat
    #: Placebo spread not explained by the estimator's own sampling model.
    excess: float = 0.0
    #: Floor contribution, in the same (absolute) units as ``excess``.
    floor_component: float = 0.0
    #: ``True`` when the design's placebo inflation is already inside the
    #: readout's standard error, so the excess must NOT be added again.
    absorbed: bool = False
    notes: tuple[str, ...] = ()

    @property
    def is_measured(self) -> bool:
        """Whether any part of the width came from the design's own evidence."""
        return bool(self.excess > 0)

    def describe(self) -> str:
        parts = [f"bias sd {self.prior.sigma:.4g} ({self.threat.threat})"]
        if self.excess > 0:
            parts.append(f"placebo excess {self.excess:.4g}")
        if self.absorbed:
            parts.append("placebo spread already inside the reported SE")
        if self.floor_component > 0:
            parts.append(f"floor {self.floor_component:.4g}")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior": self.prior.to_dict(),
            "threat": self.threat.to_dict(),
            "excess": float(self.excess),
            "floor_component": float(self.floor_component),
            "absorbed": bool(self.absorbed),
            "is_measured": bool(self.is_measured),
            "notes": list(self.notes),
            "description": self.describe(),
        }


def _design_excess_roas(
    design: dict[str, Any], se: float
) -> tuple[float, bool, list[str]]:
    """Placebo spread beyond the analytic model, on the design's ROAS scale.

    ``geo_lift_design`` reports ``se_roas`` already multiplied by
    ``calibration_factor = placebo_sd / analytic_sd`` when it had enough placebo
    windows, so the analytic value is recoverable by dividing it back out and the
    excess follows as ``se_analytic * sqrt(cf**2 - 1)``. Everything stays on the
    ROAS scale the design already reports, so no unit conversion is involved.
    """
    notes: list[str] = []
    diagnostics = design.get("diagnostics") or {}
    cf = float(diagnostics.get("calibration_factor") or 1.0)
    se_design = design.get("se_roas")
    if design.get("se_source") != "placebo_calibrated" or cf <= 1.0:
        placebo = design.get("placebo") or {}
        n_windows = int(placebo.get("n_windows") or 0)
        notes.append(
            "The design has no usable placebo distribution "
            f"({n_windows} window(s); at least 12 are needed), so the width below "
            "is the method's default floor rather than a measurement from this "
            "data."
        )
        return 0.0, False, notes
    if se_design is None or not np.isfinite(se_design) or se_design <= 0:
        notes.append(
            "The design reports no ROAS-scale standard error to compare against."
        )
        return 0.0, False, notes

    se_design = float(se_design)
    se_analytic = se_design / cf
    excess = se_analytic * math.sqrt(max(cf**2 - 1.0, 0.0))

    # Did the analyst quote the design's own (already inflated) number?
    absorbed = bool(
        np.isfinite(se) and se > 0 and abs(se - se_design) <= 0.10 * se_design
    )
    if absorbed:
        notes.append(
            "The reported standard error matches the design's placebo-calibrated "
            f"one ({se_design:.4g}), so the placebo spread is already inside it. "
            "Adding it again would inflate every calibrated interval for no "
            "reason, so only the method's floor is applied."
        )
    else:
        notes.append(
            f"Placebo spread exceeds the analytic model by {excess:.4g} on the "
            "ROAS scale — the part of the estimator's null distribution its "
            "sampling model does not explain."
        )
    return excess, absorbed, notes


def derive_bias_prior(
    *,
    value: float,
    se: float,
    estimand: str = "roas",
    design: dict[str, Any] | None = None,
    method_key: str | None = None,
    floor: float | None = None,
    explicit_sigma: float | None = None,
    explicit_mu: float = 0.0,
) -> DerivedBiasPrior:
    """Build a bias prior for one readout from its design's own evidence.

    ``explicit_sigma`` overrides everything and is the escape hatch for estimands
    whose evidence cannot be converted (see the module docstring). ``explicit_mu``
    is the only way to set a *directional* bias: the placebo machinery reports a
    spread, not a signed offset, and inferring a sign from a design would mean
    trusting an unsigned ``weekly_spend_delta`` — which inverts on a holdout.
    """
    design = design or {}
    threat = threat_for(method_key, design)
    magnitude = abs(float(value)) if np.isfinite(value) else 0.0
    notes: list[str] = []

    if explicit_sigma is not None:
        prior = BiasPrior(
            mu=float(explicit_mu),
            sigma=abs(float(explicit_sigma)),
            scale="absolute",
            correlation="shared",
            label="supplied",
            source="explicit",
        )
        return DerivedBiasPrior(
            prior=prior,
            threat=threat,
            notes=("Bias prior supplied by the caller rather than derived.",),
        )

    floor_rel = threat.floor if floor is None else float(floor)
    floor_component = floor_rel * magnitude

    excess = 0.0
    absorbed = False
    if estimand == "roas" and design:
        excess, absorbed, design_notes = _design_excess_roas(design, se)
        notes.extend(design_notes)
    elif design and estimand != "roas":
        notes.append(
            f"A design-derived width is only defined on the ROAS scale; this "
            f"readout is on the '{estimand}' scale, where the design's evidence "
            "would need a conversion whose sign and cell basis cannot be "
            "verified. Only the method's floor is applied — supply "
            "explicit_sigma to do better."
        )

    effective_excess = 0.0 if absorbed else excess
    sigma = math.hypot(effective_excess, floor_component)

    source_bits = []
    if effective_excess > 0:
        source_bits.append("placebo:excess-over-analytic")
    if absorbed:
        source_bits.append("placebo:absorbed-in-se")
    source_bits.append(f"floor:{threat.method_key}@{floor_rel:g}")
    source = "named" if not effective_excess else "+".join(source_bits)

    prior = BiasPrior(
        mu=float(explicit_mu),
        sigma=float(sigma),
        scale="absolute",
        correlation="shared",
        label=f"{threat.label} residual bias",
        source=source,
    )
    return DerivedBiasPrior(
        prior=prior,
        threat=threat,
        excess=float(excess),
        floor_component=float(floor_component),
        absorbed=absorbed,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #

#: Break-even reference per estimand. A ROAS or marginal ROAS of 1.0 pays for
#: itself; a contribution's null is zero incremental KPI.
_REFERENCE_BY_ESTIMAND = {"roas": 1.0, "mroas": 1.0, "contribution": 0.0}


@dataclass
class ExperimentSensitivityReport:
    """Whether an experiment's conclusion survives its own residual bias."""

    channel: str
    estimand: str
    value: float
    se: float
    reference: float
    sensitivity: BiasSensitivity
    derived: DerivedBiasPrior
    caveat: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def threat(self) -> DesignThreat:
        return self.derived.threat

    @property
    def verdict(self) -> str:
        return self.sensitivity.verdict

    def describe(self) -> str:
        return f"{self.sensitivity.describe()} {self.threat.caveat}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "estimand": self.estimand,
            "value": float(self.value),
            "se": float(self.se),
            "reference": float(self.reference),
            "verdict": self.verdict,
            "sensitivity": self.sensitivity.to_dict(),
            "derived": self.derived.to_dict(),
            "caveat": self.caveat,
            "notes": list(self.notes),
            "description": self.describe(),
        }

    def as_measurement_kwargs(self) -> dict[str, Any]:
        """The ``bias_*`` fields to stage onto an
        :class:`~mmm_framework.calibration.likelihood.ExperimentMeasurement`.

        The registry keeps the raw readout untouched — a measurement is what it
        is. Only the *calibration input* is adjusted, so the same readout can be
        staged under several assumed biases without rewriting history.
        """
        return {
            "bias_mu": float(self.derived.prior.mu),
            "bias_sigma": float(self.derived.prior.sigma),
            "bias_scale": "natural",
            "bias_source": self.derived.prior.source,
        }


def experiment_sensitivity(
    *,
    value: float,
    se: float,
    estimand: str = "roas",
    channel: str = "",
    design: dict[str, Any] | None = None,
    method_key: str | None = None,
    reference: float | None = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    floor: float | None = None,
    explicit_sigma: float | None = None,
    explicit_mu: float = 0.0,
    include_surface: bool = True,
) -> ExperimentSensitivityReport:
    """Price the residual bias in one experiment readout.

    Returns the tipping point — how large a bias would have to be for the
    readout to stop clearing break-even — together with the design-derived prior
    and an explicit statement of *which* threat that prior stands for.
    """
    estimand = str(estimand).lower()
    if reference is None:
        reference = _REFERENCE_BY_ESTIMAND.get(estimand, 0.0)

    derived = derive_bias_prior(
        value=value,
        se=se,
        estimand=estimand,
        design=design,
        method_key=method_key,
        floor=floor,
        explicit_sigma=explicit_sigma,
        explicit_mu=explicit_mu,
    )

    sens = bias_sensitivity_report(
        estimate=float(value),
        se=float(se),
        reference=float(reference),
        label=channel or "readout",
        reference_label=(
            "above break-even" if estimand in ("roas", "mroas") else "above zero"
        ),
        priors=[derived.prior],
        scale="absolute",
        threshold=threshold,
        include_surface=include_surface,
        units=estimand.upper(),
    )

    threat = derived.threat
    caveat = threat.caveat
    if threat.threat == "interference":
        caveat += (
            " The tipping point below therefore prices *interference and "
            "concurrent shocks*, not confounding — do not read it as a "
            "confounding sensitivity."
        )
    if not derived.is_measured:
        caveat += (
            " No part of the assumed width came from this design's own evidence; "
            "it is entirely the method's default floor."
        )

    return ExperimentSensitivityReport(
        channel=channel,
        estimand=estimand,
        value=float(value),
        se=float(se),
        reference=float(reference),
        sensitivity=sens,
        derived=derived,
        caveat=caveat,
        notes=derived.notes,
    )
