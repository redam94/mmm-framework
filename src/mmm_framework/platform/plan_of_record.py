"""What may be committed as a plan of record, and what it must carry.

#223 made a forecast *computable* with its caveats attached. This module decides
what may be **committed** — which is a different question, and the milestone's
rule forbids answering it with "disclose and commit anyway".

A committed plan is the number a variance is later computed against. Two ways
that goes wrong, and this module exists for both:

1. **The commitment is not reproducible.** A stored forecast with no model /
   spec / data fingerprint is a screenshot wearing a commitment's clothes: months
   later nobody can regenerate it, so nobody can defend the variance against it.
   Provenance is genuinely best-effort in this codebase — ``model_run`` payloads
   carry ``run_id`` but ``spec_hash`` / ``data_fingerprint`` are stamped only by
   the agent host path inside a try/except, and the fingerprint returns ``None``
   once the dataset file is gone. So committing without resolvable provenance is
   **refused, naming what is missing**, rather than committed with blanks.

2. **The commitment is curve fiction.** A plan that pushes a channel past the
   spend the model observed is asking the saturation curve about a region with
   no data. #223 flags that; committing must *refuse* it rather than inherit the
   flag, because a variance against a fictional number is meaningless.

Three refusals, each derived from a field the forecast already computes rather
than from a new judgement:

* **horizon vs trend policy** — a held-flat spline/GP/piecewise trend produces an
  interval that provably does not widen with horizon. That is not an uncertainty
  statement, and committing to a long horizon on one is committing to a band that
  is decorative past the first few periods.
* **residual autocorrelation** — Ljung-Box rejecting means the interval is
  knowingly too narrow. #223 states it; commit-time refuses it.
* **spend beyond observed support** — per channel, from the forecast's own
  ``extrapolated_channels``.

Each is overridable, and an override is **recorded in the committed payload**, so
a reader of the commitment sees which gate was waived and by whom. An override
that is not written down is indistinguishable from a gate that never fired.

Lean-core: standard library only. Imported by the server and the agent alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CommitRefusal",
    "Committability",
    "DEFAULT_FLEXIBLE_TREND_HORIZON_CAP",
    "assess_committability",
    "build_commit_payload",
    "provenance_gaps",
]

#: Periods beyond which a held-flat trend's interval stops meaning anything.
#: Deliberately short: the band does not widen AT ALL under that policy, so the
#: only honest window is one where the trend's own drift is negligible.
DEFAULT_FLEXIBLE_TREND_HORIZON_CAP = 13

#: Gate identifiers, so an override names a specific refusal rather than
#: blanket-waiving everything.
GATE_TREND_HORIZON = "trend_horizon"
GATE_RESIDUAL_AUTOCORRELATION = "residual_autocorrelation"
GATE_SPEND_SUPPORT = "spend_support"
GATE_INTERVAL_AVAILABLE = "interval_available"

ALL_GATES = (
    GATE_TREND_HORIZON,
    GATE_RESIDUAL_AUTOCORRELATION,
    GATE_SPEND_SUPPORT,
    GATE_INTERVAL_AVAILABLE,
)


@dataclass(frozen=True)
class CommitRefusal:
    """One reason a forecast may not be committed as it stands."""

    gate: str
    reason: str
    #: What the caller would have to change, as opposed to waive.
    remedy: str

    def to_dict(self) -> dict[str, str]:
        return {"gate": self.gate, "reason": self.reason, "remedy": self.remedy}


@dataclass
class Committability:
    """Whether this forecast may be committed, and what stands in the way."""

    committable: bool
    refusals: list[CommitRefusal] = field(default_factory=list)
    #: Gates the caller explicitly waived, recorded into the payload.
    overrides: dict[str, str] = field(default_factory=dict)
    #: Provenance fields that could not be resolved.
    missing_provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "committable": self.committable,
            "refusals": [r.to_dict() for r in self.refusals],
            "overrides": dict(self.overrides),
            "missing_provenance": list(self.missing_provenance),
        }

    def blocking_gates(self) -> list[str]:
        return [r.gate for r in self.refusals]


def provenance_gaps(provenance: dict[str, Any] | None) -> list[str]:
    """Which of ``run_id`` / ``spec_hash`` / ``data_fingerprint`` are unresolved.

    ``model_path`` is included because auto-save failure is a caught, non-fatal
    branch at fit time: a run can exist with no model on disk, and a commitment
    that cannot reload its model cannot be reproduced.
    """
    p = provenance or {}
    required = ("run_id", "spec_hash", "data_fingerprint", "model_path")
    return [k for k in required if not p.get(k)]


def assess_committability(
    forecast: dict[str, Any] | None,
    *,
    provenance: dict[str, Any] | None = None,
    valuation: dict[str, Any] | None = None,
    overrides: dict[str, str] | None = None,
    horizon_cap: int = DEFAULT_FLEXIBLE_TREND_HORIZON_CAP,
) -> Committability:
    """Decide whether ``forecast`` may become a plan of record.

    ``forecast`` is the payload from the ``forecast_plan`` op (#223).
    ``overrides`` maps a gate id to the acknowledgement text; an overridden gate
    stops blocking but is recorded. Unresolvable provenance is NOT overridable —
    waiving it would produce exactly the unreproducible commitment this module
    exists to prevent.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if k in ALL_GATES and v}
    refusals: list[CommitRefusal] = []

    if not forecast:
        return Committability(
            committable=False,
            refusals=[
                CommitRefusal(
                    gate="forecast",
                    reason=(
                        "There is no forecast to commit. A plan of record is the "
                        "KPI level a variance is later measured against, and an "
                        "allocation alone does not state one."
                    ),
                    remedy="Run a forecast under this plan first.",
                )
            ],
            overrides=overrides,
            missing_provenance=provenance_gaps(provenance),
        )

    fields = forecast.get("caveat_fields") or {}
    n_periods = len(forecast.get("periods") or [])

    # 1. Horizon vs trend policy.
    widens = fields.get("interval_widens_with_horizon")
    policy = (fields.get("trend_extrapolation") or {}).get("policy")
    if widens is False and n_periods > horizon_cap:
        refusals.append(
            CommitRefusal(
                gate=GATE_TREND_HORIZON,
                reason=(
                    f"The trend is {policy}, so the interval does not widen with "
                    f"horizon — week {n_periods} is shown as no less certain than "
                    f"week 1, which cannot be true. Committing to a "
                    f"{n_periods}-period window on that basis commits to a band "
                    f"that is decorative past period {horizon_cap}."
                ),
                remedy=(
                    f"Commit at most {horizon_cap} periods, or re-fit with a "
                    "linear trend, which extrapolates in closed form."
                ),
            )
        )

    # 2. Residual autocorrelation.
    ra = fields.get("residual_autocorrelation") or {}
    if ra.get("autocorrelated"):
        p = ra.get("ljung_box_p")
        refusals.append(
            CommitRefusal(
                gate=GATE_RESIDUAL_AUTOCORRELATION,
                reason=(
                    f"The fitted residuals are autocorrelated (Ljung-Box "
                    f"p={p:.3g}) while the predictive noise is iid, so this "
                    "interval is knowingly too narrow. A variance measured "
                    "against it will flag as significant more often than it "
                    "should."
                ),
                remedy=(
                    "Address the autocorrelation (a missing seasonal or trend "
                    "term is the usual cause), or commit the point estimate "
                    "with this gate explicitly waived."
                ),
            )
        )

    # 3. Spend beyond observed support.
    extrapolated = fields.get("extrapolated_channels") or []
    if extrapolated:
        names = ", ".join(
            f"{c.get('channel')} ({float(c.get('multiple', 0)):.2f}x)"
            for c in extrapolated
        )
        refusals.append(
            CommitRefusal(
                gate=GATE_SPEND_SUPPORT,
                reason=(
                    f"The plan funds {names} past the spend the model observed. "
                    "The saturation curve has no data there, so the committed "
                    "number is curve fiction and a variance against it measures "
                    "nothing."
                ),
                remedy=(
                    "Bring the plan inside observed spend, or run an experiment "
                    "at the proposed level before committing to it."
                ),
            )
        )

    # 4. No interval at all (a single-draw approximate posterior).
    if fields.get("interval_available") is False:
        refusals.append(
            CommitRefusal(
                gate=GATE_INTERVAL_AVAILABLE,
                reason=(
                    "This forecast has no interval — the posterior has too few "
                    "draws to form one, which is what an approximate (MAP/ADVI) "
                    "fit produces. Committing a point estimate as a plan of "
                    "record states a precision the fit does not have."
                ),
                remedy="Re-fit with NUTS before committing.",
            )
        )

    blocking = [r for r in refusals if r.gate not in overrides]
    missing = provenance_gaps(provenance)

    # Provenance is not overridable. A commitment that cannot be regenerated is
    # the failure this module exists to prevent, and waiving it would reintroduce
    # exactly that under a different name.
    if missing:
        blocking = blocking + [
            CommitRefusal(
                gate="provenance",
                reason=(
                    "Cannot resolve the provenance this commitment would need to "
                    f"be reproducible: {', '.join(missing)}. Without it the "
                    "committed number cannot be regenerated, so a later variance "
                    "against it cannot be defended."
                ),
                remedy=(
                    "Re-fit and save the model so the run carries its spec hash, "
                    "data fingerprint and model path."
                ),
            )
        ]

    if valuation is not None and not valuation.get("value_per_kpi"):
        blocking = blocking + [
            CommitRefusal(
                gate="valuation",
                reason=(
                    "This plan is denominated in dollars but no KPI valuation "
                    "resolved, so one KPI unit would be assumed to be worth one "
                    "dollar."
                ),
                remedy="Set the project's economics preference, or pass one.",
            )
        ]

    return Committability(
        committable=not blocking,
        refusals=blocking,
        overrides=overrides,
        missing_provenance=missing,
    )


def build_commit_payload(
    *,
    forecast: dict[str, Any],
    allocation: list[dict[str, Any]] | None = None,
    flighting: dict[str, Any] | None = None,
    calendar: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    valuation: dict[str, Any] | None = None,
    objective: dict[str, Any] | None = None,
    committability: Committability | None = None,
) -> dict[str, Any]:
    """The frozen payload of a committed version.

    Everything a later reader needs to (a) know what was promised and (b)
    regenerate it. The forecast is stored WHOLE — including its thinned base64
    per-period draws — because a window-total interval cannot be recovered from
    per-period bounds, and variance work grades against draws.
    """
    return {
        "schema_version": 1,
        "forecast": forecast,
        "allocation": list(allocation or []),
        "flighting": flighting,
        "calendar": calendar,
        "provenance": dict(provenance or {}),
        "valuation": dict(valuation or {}),
        "objective": dict(objective or {}),
        # Recorded, not inferred: a reader sees which gate was waived rather
        # than having to re-derive whether one fired at all.
        "committability": (
            committability.to_dict()
            if committability is not None
            else {"committable": True, "refusals": [], "overrides": {}}
        ),
    }
