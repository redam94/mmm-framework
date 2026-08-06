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
    "ReproductionResult",
    "assess_committability",
    "build_commit_payload",
    "provenance_gaps",
    "reproduce_committed_plan",
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
    plan_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen payload of a committed version.

    Everything a later reader needs to (a) know what was promised and (b)
    regenerate it. The forecast is stored WHOLE — including its thinned base64
    per-period draws — because a window-total interval cannot be recovered from
    per-period bounds, and variance work grades against draws.
    """
    return {
        "schema_version": 1,
        # The working plan's own payload, frozen verbatim (#225 remainder):
        # pacing joins delivery against `plan_payload`, so a committed version
        # must carry the same shape the draft did or the retarget in
        # `latest_budget_plan_for_project` would hand pacing a payload it
        # cannot read.
        "plan_payload": dict(plan_payload or {}),
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


# ---------------------------------------------------------------------------
# Reproduction
#
# The claim a commitment makes is not "here is a number" but "here is a number
# anyone can regenerate". This is where that claim is checked. Heavy imports are
# function-local so the module stays lean-core — the gate above is imported by
# the server and the agent, neither of which should pull PyMC to read it.
# ---------------------------------------------------------------------------


@dataclass
class ReproductionResult:
    """Whether a committed plan regenerates from its own provenance."""

    reproduced: bool
    #: Set when reproduction could not even be ATTEMPTED (missing model, changed
    #: data). Distinct from a mismatch: "I refuse to check" and "I checked and it
    #: differs" are different statements about a commitment.
    refused: bool
    reason: str | None = None
    max_abs_diff: float | None = None
    tolerance: float = 1e-9
    #: What moved, when reproduction ran and disagreed.
    diffs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reproduced": self.reproduced,
            "refused": self.refused,
            "reason": self.reason,
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
            "diffs": dict(self.diffs),
        }


def reproduce_committed_plan(
    version: dict[str, Any],
    *,
    tolerance: float = 1e-9,
    models_dir: str | None = None,
) -> ReproductionResult:
    """Regenerate a committed plan's forecast from its recorded provenance.

    Reloads the model from the run's saved directory, rebuilds the panel from the
    **saved** run's spec (not any current session spec, which may have been
    edited since), and re-runs the forecast with the recorded plan and seed. The
    recomputed mean and interval bounds must match the stored snapshot.

    **Refuses rather than reports a mismatch** when the inputs are not the ones
    that were committed — a changed dataset, a missing model directory. Those are
    different statements: "this commitment no longer reproduces" would blame the
    model for a moved file.
    """
    import numpy as np

    payload = version.get("payload") or {}
    snapshot = payload.get("forecast") or {}
    prov = payload.get("provenance") or {}

    gaps = provenance_gaps(prov)
    if gaps:
        return ReproductionResult(
            reproduced=False,
            refused=True,
            reason=(
                "Cannot reproduce: the commitment is missing "
                f"{', '.join(gaps)}. It should not have been committable."
            ),
            tolerance=tolerance,
        )

    import os

    model_path = prov.get("model_path")
    if models_dir:
        model_path = os.path.join(models_dir, os.path.basename(str(model_path)))
    if not model_path or not os.path.exists(model_path):
        return ReproductionResult(
            reproduced=False,
            refused=True,
            reason=(
                f"Cannot reproduce: the saved model at {model_path!r} is gone, "
                "so the committed number cannot be regenerated."
            ),
            tolerance=tolerance,
        )

    # The dataset must be the one that was committed against. A silent
    # re-fingerprint mismatch is the difference between "the model drifted" and
    # "somebody replaced the data".
    from .runs import data_fingerprint

    recorded = prov.get("data_fingerprint")
    recorded_fp = recorded.get("md5") if isinstance(recorded, dict) else recorded
    # The panel has to be rebuilt from the dataset, so its path is part of what
    # makes a commitment reproducible — recorded explicitly, else carried on the
    # fingerprint dict that `data_fingerprint()` returns.
    dataset_path = prov.get("dataset_path") or (
        recorded.get("path") if isinstance(recorded, dict) else None
    )
    if not dataset_path:
        return ReproductionResult(
            reproduced=False,
            refused=True,
            reason=(
                "Cannot reproduce: the commitment records no dataset path, so "
                "the panel cannot be rebuilt as it was fitted."
            ),
            tolerance=tolerance,
        )
    if dataset_path:
        current = data_fingerprint(dataset_path)
        current_md5 = (current or {}).get("md5")
        if current_md5 is None:
            return ReproductionResult(
                reproduced=False,
                refused=True,
                reason=(
                    f"Cannot reproduce: the dataset at {dataset_path!r} is no "
                    "longer readable."
                ),
                tolerance=tolerance,
            )
        if current_md5 != recorded_fp:
            return ReproductionResult(
                reproduced=False,
                refused=True,
                reason=(
                    "Cannot reproduce: the dataset behind this commitment has "
                    f"changed (committed {recorded_fp}, now {current_md5}). The "
                    "committed number was correct for the data it was made on; "
                    "recomputing it against different data would not verify it."
                ),
                tolerance=tolerance,
            )

    try:
        from mmm_framework.agents.fitting import build_model, saved_model_settings
        from mmm_framework.planning.forecast import forecast_under_plan
        from mmm_framework.serialization import MMMSerializer

        saved = saved_model_settings(model_path)
        saved_spec = saved.get("spec")
        if not saved_spec:
            return ReproductionResult(
                reproduced=False,
                refused=True,
                reason=(
                    "Cannot reproduce: the saved run carries no model spec, so "
                    "the panel cannot be rebuilt as it was fitted."
                ),
                tolerance=tolerance,
            )
        # Rebuild from the SAVED spec — a current session spec may have been
        # edited since the fit, and the serializer validates the panel against
        # what the model was actually trained on.
        rebuilt = build_model(saved_spec, str(dataset_path))
        panel = getattr(rebuilt, "panel", None)
        model = MMMSerializer.load(model_path, panel)

        plan = payload.get("plan_media") or snapshot.get("plan_media")
        controls = payload.get("plan_controls")
        seed = payload.get("random_seed", 42)
        if not plan:
            return ReproductionResult(
                reproduced=False,
                refused=True,
                reason=(
                    "Cannot reproduce: the commitment records no per-period "
                    "spend plan, so there is nothing to re-forecast."
                ),
                tolerance=tolerance,
            )
        recomputed = forecast_under_plan(
            model,
            plan,
            future_controls=controls,
            interval=float(snapshot.get("interval", 0.9)),
            max_draws=int(snapshot.get("n_draws", 200)),
            random_seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        return ReproductionResult(
            reproduced=False,
            refused=True,
            reason=f"Cannot reproduce: {exc}",
            tolerance=tolerance,
        )

    diffs: dict[str, float] = {}
    for key, got in (
        ("mean", recomputed.mean),
        ("lower", recomputed.lower),
        ("upper", recomputed.upper),
    ):
        want = np.asarray(snapshot.get(key) or [], dtype=float)
        have = np.asarray(got, dtype=float)
        if want.shape != have.shape:
            return ReproductionResult(
                reproduced=False,
                refused=False,
                reason=(
                    f"Recomputed {key} has {have.shape} periods against the "
                    f"committed {want.shape}."
                ),
                tolerance=tolerance,
            )
        both_nan = np.isnan(want) & np.isnan(have)
        d = np.abs(np.where(both_nan, 0.0, want - have))
        diffs[key] = float(np.nanmax(d)) if d.size else 0.0

    worst = max(diffs.values()) if diffs else 0.0
    return ReproductionResult(
        reproduced=bool(worst <= tolerance),
        refused=False,
        reason=None if worst <= tolerance else "Recomputed values differ.",
        max_abs_diff=worst,
        tolerance=tolerance,
        diffs=diffs,
    )
