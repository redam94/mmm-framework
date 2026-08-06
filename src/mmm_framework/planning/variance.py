"""Variance to plan: delivery-driven vs unexplained, with a bridge that closes
(issue #227).

Without a refit, only two buckets are identifiable, and this module ships
exactly those:

1. **Delivery variance** — ``g_plan(S_actual) − g_plan(S_plan)`` per channel,
   holding the committed model fixed and re-running
   :func:`~mmm_framework.planning.forecast.forecast_under_plan` on actual
   spend: a clean counterfactual on the committed posterior, with a paired-draw
   interval on the total.
2. **Unexplained** — ``actual KPI − forecast-under-actual-spend``, which mixes
   baseline movement, competitor action, data error, model error and noise,
   and is LABELLED for what it contains rather than attributed.

**The refit "effectiveness" split is refused.** ``g_new(S_actual) −
g_plan(S_actual)`` is a difference between two posteriors absorbing more data,
a different training window, a changed spec and MC noise; labelling that
"effectiveness variance" is a manufactured causal claim from a subtraction.
The word "effectiveness" appears on no row and in no column header here, and a
``refit_run_id`` gets a stated refusal (the platform wrapper attaches the
``compare_runs`` diff — what CHANGED between the runs — instead).

The residual is scored against the committed interval BEFORE it is scored as a
variance: a miss inside the committed band is "within the committed
uncertainty", not something to explain. That needs the committed per-period
draws — a window-total interval cannot be recovered from per-period bounds —
which is why the #225 payload stores them whole.

Rows sum to the actual-minus-committed gap **exactly** (1e-9), by
construction: the forecast's own decomposition closes (mean = Σ by_channel +
baseline), so per-channel deltas + the baseline delta reproduce the
forecast-to-forecast gap bit-for-bit, and the unexplained line is defined as
the remainder to actuals. Supplied adjustment lines subtract from the
unexplained remainder, never from a channel — see :func:`supplied_line`.

Lean: numpy + the finance vocabulary. The store-reading wrapper is
:mod:`mmm_framework.platform.variance`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..finance.lines import MODELLED, RESIDUAL, SUPPLIED, BridgeLine

__all__ = [
    "VarianceBridge",
    "supplied_line",
    "variance_to_plan",
]

REFIT_REFUSAL = (
    "A refit-based split is refused: the difference between a new posterior "
    "and the committed one mixes more data, a different training window, any "
    "spec changes and Monte Carlo noise — labelling that subtraction a "
    "variance would manufacture a causal claim. What CAN be said is what "
    "changed between the runs; see the attached run diff."
)


def _decode_draws(draws_b64: str, n_draws: int, n_periods: int) -> np.ndarray:
    """The #225 snapshot convention: base64, little-endian float32,
    ``(n_draws, n_periods)`` row-major."""
    raw = base64.b64decode(draws_b64)
    return np.frombuffer(raw, dtype="<f4").reshape(int(n_draws), int(n_periods))


def supplied_line(
    name: str,
    value: float,
    *,
    source_note: str,
    channel: str | None = None,
    kpi_kind_is_dollar: bool = True,
) -> BridgeLine:
    """A human-supplied adjustment line (gross-to-net, returns, trade spend).

    Rides the #220 provenance vocabulary: a point value, a REQUIRED source
    note, no interval fields (both enforced by :class:`BridgeLine` itself).
    Two refusals live here because they are about what a supplied line may
    CLAIM, not about its shape:

    * ``channel`` must be ``None`` — a supplied adjustment maps a TOTAL only.
      Rescaling per-channel contributions would produce a net-scaled ROI the
      model never estimated, wearing the model's label.
    * a non-dollar KPI is refused — netting adjustments denominate in money,
      and applying one to a units/index KPI silently changes what the bridge
      is a bridge OF.
    """
    if channel is not None:
        raise ValueError(
            f"supplied line {name!r}: a supplied adjustment maps the TOTAL "
            f"only; asking it to restate channel {channel!r} would produce a "
            "net-scaled per-channel number the model never estimated."
        )
    if not kpi_kind_is_dollar:
        raise ValueError(
            f"supplied line {name!r}: the KPI is not dollar-denominated, so a "
            "monetary netting adjustment cannot be applied to it without "
            "changing what the bridge measures. Declare a valuation "
            "(kind + margin/price) first."
        )
    return BridgeLine(
        name=name,
        value=float(value),
        provenance=SUPPLIED,
        source_note=source_note,
    )


@dataclass(frozen=True)
class VarianceBridge:
    """The two-bucket bridge from the committed forecast to realized KPI.

    ``rows`` are the addends: per-channel delivery lines + the baseline
    delivery line + any supplied lines + the unexplained remainder. They sum
    to ``actual_kpi − committed_kpi`` to 1e-9. ``within_committed_interval``
    LEADS every rendering: a miss inside the committed band is within the
    committed uncertainty, not a gap demanding a story.
    """

    committed_kpi: float
    actual_kpi: float
    gap: float
    rows: list[BridgeLine]
    #: Whether the realized total landed inside the committed window-total
    #: interval, computed from the committed per-period DRAWS.
    within_committed_interval: bool | None
    committed_lower: float | None
    committed_upper: float | None
    interval_mass: float | None
    #: Total delivery variance with its paired-draw interval.
    delivery_total: float
    delivery_lower: float | None
    delivery_upper: float | None
    unexplained: float
    #: The period labels this bridge summed — stated, because a ragged panel's
    #: coverage differs per geo and "the window" is otherwise a guess.
    period_set: list[str]
    #: KPI-units → dollars conversion, when a valuation resolved. ``None``
    #: suppresses every dollar figure rather than inventing one.
    value_per_kpi: float | None
    value_source: str | None
    #: True when any channel is efficiency-measured: the blended dollar
    #: headline is suppressed (a blended $ over mixed divisors is meaningless).
    dollar_headline_suppressed: bool = False
    caveats: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    @property
    def closes(self) -> bool:
        total = sum(ln.value for ln in self.rows)
        return abs(total - self.gap) <= max(abs(self.gap) * 1e-9, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "committed_kpi": self.committed_kpi,
            "actual_kpi": self.actual_kpi,
            "gap": self.gap,
            "rows": [ln.to_dict() for ln in self.rows],
            "rows_dollars": (
                None
                if self.value_per_kpi is None or self.dollar_headline_suppressed
                else {ln.name: ln.value * self.value_per_kpi for ln in self.rows}
            ),
            "within_committed_interval": self.within_committed_interval,
            "committed_lower": self.committed_lower,
            "committed_upper": self.committed_upper,
            "interval_mass": self.interval_mass,
            "delivery_total": self.delivery_total,
            "delivery_lower": self.delivery_lower,
            "delivery_upper": self.delivery_upper,
            "unexplained": self.unexplained,
            "period_set": list(self.period_set),
            "value_per_kpi": self.value_per_kpi,
            "value_source": self.value_source,
            "dollar_headline_suppressed": self.dollar_headline_suppressed,
            "closes": self.closes,
            "caveats": list(self.caveats),
            "refusals": list(self.refusals),
        }


def variance_to_plan(
    model: Any,
    committed_payload: dict[str, Any],
    actual_media: dict[str, list[float]],
    actuals: list[dict[str, Any]],
    *,
    supplied: list[BridgeLine] | None = None,
    value_per_kpi: float | None = None,
    value_source: str | None = None,
    refit_run_id: str | None = None,
    channel_meta: dict[str, bool] | None = None,
    fit_diagnostics: dict[str, Any] | None = None,
) -> VarianceBridge:
    """Build the two-bucket variance bridge against a committed plan.

    Parameters
    ----------
    model:
        The COMMITTED model, reloaded from the version's provenance — the
        platform wrapper does that reload; passing a newer fit here would
        silently turn bucket 1 into the refused refit comparison.
    committed_payload:
        The committed version's ``payload`` (#225): its ``forecast`` snapshot
        supplies the periods, the interval, the per-period draws and the plan
        the forecast ran on.
    actual_media:
        ``{channel: [per-period actual spend]}`` over the committed periods —
        from the delivery store, label-joined by the wrapper.
    actuals:
        Latest-as-of realized-KPI rows (``{period, kpi_value}``). Every
        committed period must be covered — a bridge over a partially-realized
        window would compare a full-window commitment against a part-window
        actual, so it REFUSES instead.
    supplied:
        :func:`supplied_line` adjustments. They subtract from the unexplained
        remainder (never from a channel), so the bridge still closes.
    refit_run_id:
        Refused with :data:`REFIT_REFUSAL`; the wrapper attaches a run diff.
    channel_meta:
        ``{channel: is_monetary}`` from ``resolve_channel_divisor``; any
        ``False`` suppresses the blended dollar headline.
    fit_diagnostics:
        The committed fit's diagnostics. When ``at_boundary`` names pinned
        columns (a ``sum_equals``-constrained frequentist fit), the
        "independent reconciliation" framing is suppressed — the gap is zero
        by construction there and would read as corroboration.
    """
    from .forecast import forecast_under_plan

    snapshot = committed_payload.get("forecast") or {}
    periods = [str(p) for p in (snapshot.get("periods") or [])]
    if not periods:
        raise ValueError(
            "The committed payload carries no forecast periods; there is "
            "nothing to bridge against."
        )
    plan_media = committed_payload.get("plan_media") or snapshot.get("plan_media")
    if not plan_media:
        raise ValueError(
            "The committed payload records no per-period spend plan "
            "(plan_media); delivery variance cannot be computed without the "
            "plan the forecast ran on."
        )
    plan_controls = committed_payload.get("plan_controls") or snapshot.get(
        "plan_controls"
    )
    seed = int(
        committed_payload.get("random_seed")
        or snapshot.get("random_seed")
        or (committed_payload.get("provenance") or {}).get("random_seed")
        or 42
    )
    interval = float(snapshot.get("interval", 0.9))
    max_draws = int(snapshot.get("n_draws", 200))

    caveats: list[str] = []
    refusals: list[str] = []
    if refit_run_id:
        refusals.append(REFIT_REFUSAL)

    # -- realized KPI over the committed window ---------------------------
    by_period = {str(r.get("period")): float(r.get("kpi_value")) for r in actuals}
    missing = [p for p in periods if p not in by_period]
    if missing:
        raise ValueError(
            "The fiscal window is not fully covered by actuals: "
            f"{len(missing)} of {len(periods)} committed periods have no "
            f"realized KPI (first missing: {missing[0]}). A bridge over a "
            "partially-realized window would compare a full-window "
            "commitment against a part-window actual — upload the missing "
            "periods or wait for them."
        )
    actual_kpi = float(sum(by_period[p] for p in periods))

    # -- the committed number and its own interval, from the DRAWS --------
    committed_kpi = float(np.sum(np.asarray(snapshot.get("mean"), dtype=float)))
    within = c_lo = c_hi = None
    try:
        draws = _decode_draws(snapshot["draws_b64"], snapshot["n_draws"], len(periods))
        totals = draws.sum(axis=1)
        if totals.shape[0] >= 10:
            lo_q = (1 - interval) / 2 * 100
            c_lo = float(np.percentile(totals, lo_q))
            c_hi = float(np.percentile(totals, 100 - lo_q))
            within = bool(c_lo <= actual_kpi <= c_hi)
        else:
            caveats.append(
                "The committed snapshot has too few draws for a window-total "
                "interval; the within-interval verdict is unavailable, not "
                "passed."
            )
    except Exception:  # noqa: BLE001 — verdict unavailable, said explicitly
        caveats.append(
            "The committed per-period draws could not be decoded; the "
            "within-interval verdict is unavailable, not passed."
        )

    # -- bucket 1: delivery variance on the COMMITTED posterior -----------
    # Two paired forecasts (same seed, same thinning): the plan as committed,
    # and the plan actually delivered. Per-channel rows from the forecast's
    # own decomposition; the paired per-draw totals give the interval.
    fc_plan = forecast_under_plan(
        model,
        plan_media,
        future_controls=plan_controls,
        interval=interval,
        max_draws=max_draws,
        random_seed=seed,
    )
    fc_actual = forecast_under_plan(
        model,
        actual_media,
        future_controls=plan_controls,
        interval=interval,
        max_draws=max_draws,
        random_seed=seed,
    )

    rows: list[BridgeLine] = []
    for ch in model.channel_names:
        d = float(
            np.sum(np.asarray(fc_actual.by_channel[ch], dtype=float))
            - np.sum(np.asarray(fc_plan.by_channel[ch], dtype=float))
        )
        rows.append(
            BridgeLine(
                name=f"Delivery — {ch}",
                value=d,
                provenance=MODELLED,
                basis="committed posterior, paired counterfactual",
                note=(
                    "What the committed model says the spend divergence on "
                    "this channel was worth. Point value from the posterior "
                    "mean decomposition; the interval lives on the total."
                ),
            )
        )
    baseline_delta = float(
        np.sum(np.asarray(fc_actual.baseline, dtype=float))
        - np.sum(np.asarray(fc_plan.baseline, dtype=float))
    )
    if abs(baseline_delta) > 1e-12:
        rows.append(
            BridgeLine(
                name="Delivery — baseline interaction",
                value=baseline_delta,
                provenance=MODELLED,
                basis="committed posterior, paired counterfactual",
                note=(
                    "Non-media terms differing between the two counterfactuals "
                    "(controls held at the committed assumption)."
                ),
            )
        )

    delivery_total = float(
        np.sum(np.asarray(fc_actual.mean, dtype=float))
        - np.sum(np.asarray(fc_plan.mean, dtype=float))
    )
    d_lo = d_hi = None
    try:
        delta_draws = fc_actual.draws().sum(axis=1) - fc_plan.draws().sum(axis=1)
        if delta_draws.shape[0] >= 10:
            lo_q = (1 - interval) / 2 * 100
            d_lo = float(np.percentile(delta_draws, lo_q))
            d_hi = float(np.percentile(delta_draws, 100 - lo_q))
    except Exception:  # noqa: BLE001
        pass

    # -- exactness: the re-run of the committed plan vs the snapshot ------
    # The identity Σrows == actual − committed relies on fc_plan reproducing
    # the committed mean bit-for-bit (same model, seed, thinning — the #225
    # reproduction guarantee). When it does not, the difference is a real
    # quantity with a name, and carrying it as its own row keeps the closure
    # EXACT algebraically instead of "usually within tolerance".
    fc_plan_total = float(np.sum(np.asarray(fc_plan.mean, dtype=float)))
    drift = fc_plan_total - committed_kpi
    if abs(drift) > max(1e-6 * abs(committed_kpi), 1e-6):
        # Not numerical noise: the model this bridge was handed does not
        # reproduce the committed forecast, so its "delivery variance" would
        # be a different posterior's opinion wearing the committed label —
        # exactly the refit comparison this module refuses.
        raise ValueError(
            "The supplied model does not reproduce the committed forecast "
            f"(re-run of the committed plan differs by {drift:+.6g} from the "
            f"committed {committed_kpi:.6g}). Refusing to build the bridge: "
            "load the committed run — a delivery bucket computed on a "
            "different posterior would be the refused refit comparison in "
            "disguise."
        )
    if abs(drift) > 1e-12:
        rows.append(
            BridgeLine(
                name="Reproduction drift",
                value=drift,
                provenance=RESIDUAL,
                note=(
                    "Re-run of the committed plan vs the committed snapshot. "
                    "Sub-tolerance numerical drift, carried as its own row so "
                    "the bridge closes exactly instead of 'within tolerance'."
                ),
            )
        )
        caveats.append(
            f"Re-running the committed plan drifted {drift:+.3g} from the "
            "committed snapshot (numerical, below the refusal threshold); "
            "the drift is carried as its own row."
        )

    # -- supplied lines + bucket 2: the labelled remainder ----------------
    supplied = list(supplied or [])
    for ln in supplied:
        if ln.provenance is not SUPPLIED:
            raise ValueError(
                f"supplied line {ln.name!r} carries provenance "
                f"{ln.provenance.value!r}; only SUPPLIED lines may be passed "
                "here — a 'modelled' adjustment would be a model claim "
                "nobody modelled."
            )
    rows.extend(supplied)
    supplied_total = float(sum(ln.value for ln in supplied))

    unexplained = (
        actual_kpi
        - float(np.sum(np.asarray(fc_actual.mean, dtype=float)))
        - supplied_total
    )
    rows.append(
        BridgeLine(
            name="Unexplained",
            value=unexplained,
            provenance=RESIDUAL,
            note=(
                "Actual KPI minus the committed model's forecast under the "
                "spend actually delivered"
                + (" minus supplied adjustments" if supplied else "")
                + ". Mixes baseline movement, competitor action, data error, "
                "model error and noise — labelled, not attributed."
            ),
        )
    )

    # -- framing gates ----------------------------------------------------
    at_boundary = list((fit_diagnostics or {}).get("at_boundary") or [])
    if at_boundary:
        caveats.append(
            "This fit pins column(s) "
            + ", ".join(str(b) for b in at_boundary)
            + " at a sum_equals constraint boundary, so parts of the bridge "
            "close by construction rather than by measurement — do not read "
            "the closure as independent reconciliation."
        )
    meta = dict(channel_meta or {})
    suppress_dollars = any(v is False for v in meta.values())
    if suppress_dollars and value_per_kpi is not None:
        caveats.append(
            "The portfolio mixes efficiency-measured channels, so the blended "
            "dollar figures are suppressed rather than invented."
        )
    if within is False:
        caveats.append(
            "The realized total landed OUTSIDE the committed interval: the "
            "miss exceeds the uncertainty that was committed to."
        )
    elif within is True:
        caveats.append(
            "The realized total is WITHIN the committed interval — this "
            "bridge explains composition, not a surprise."
        )

    return VarianceBridge(
        committed_kpi=committed_kpi,
        actual_kpi=actual_kpi,
        gap=actual_kpi - committed_kpi,
        rows=rows,
        within_committed_interval=within,
        committed_lower=c_lo,
        committed_upper=c_hi,
        interval_mass=interval if c_lo is not None else None,
        delivery_total=delivery_total,
        delivery_lower=d_lo,
        delivery_upper=d_hi,
        unexplained=unexplained,
        period_set=periods,
        value_per_kpi=value_per_kpi,
        value_source=value_source,
        dollar_headline_suppressed=suppress_dollars,
        caveats=caveats,
        refusals=refusals,
    )
