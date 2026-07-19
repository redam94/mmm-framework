"""Net experiment economics — test revenue loss vs reallocation gain, netted.

Both halves already exist: `opportunity_cost.compute_opportunity_cost` prices
the short-term cost of deviating from BAU during the test, and `evoi` values the
learning to the budget decision (KPI-contribution units, capped by EVPI). They
were only bridged by a unitless learning-to-cost ratio. This module returns the
single decision figure the client weighs:

    net value = E[reallocation gain the learning unlocks over the horizon]
              − E[profit given up while the test runs]

in DOLLARS (when a margin resolves; KPI units otherwise), with a distribution
(reusing the opportunity-cost module's per-draw KPI deltas — no new posterior
passes), ``P(net > 0)``, and the break-even horizon.

Two honesty devices:

* **Decay haircut** — EVOI values a posterior that stays sharp forever; channel
  information decays with a half-life (`eig.channel_half_life`). The gain is
  the naive horizon-total × the average informativeness retained over the
  horizon, ``(1/H) Σ_w 0.5^(w/h) · disc^w`` — always ≤ 1.
* **EVPI cap** — the gain can never exceed the value of PERFECT information.

The signed ``net_profit`` convention keeps a money-saving holdout coherent: its
"loss" is negative (the saved spend outweighs the forgone margin), so the net
value exceeds the gain and the break-even horizon is 0.

numpy only (kernel-safe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class ExperimentNetValue:
    """The netted decision figure. ``unit`` is '$' when a margin resolved, else
    'KPI units' (spend cannot be netted against KPI — noted in assumptions)."""

    channel: str
    unit: str  # '$' | 'KPI units'
    basis: str  # 'model_anchored' | 'evoi_bounded' | 'insufficient'
    # loss side (positive = money/KPI given up during the test)
    test_loss: float | None
    test_loss_p5: float | None
    test_loss_p95: float | None
    net_profit_during_test: float | None  # signed E[impact of running the test]
    # gain side
    evoi_raw: float | None  # KPI units, as computed
    evpi_cap: float | None
    decay_factor: float | None  # avg informativeness retained over the horizon
    reallocation_gain: float | None  # decayed, capped, in `unit`
    # net
    net_value: float | None
    net_value_p5: float | None
    net_value_p95: float | None
    prob_net_positive: float | None
    breakeven_horizon_weeks: float | None  # 0 = immediately net-positive
    # provenance
    horizon_weeks: int
    half_life_weeks: float | None
    margin_per_kpi: float | None
    assumptions: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _clean(v: Any) -> Any:
            if isinstance(v, float):
                return float(v) if math.isfinite(v) else None
            return v

        return {k: _clean(v) for k, v in self.__dict__.items()}


def _decay_weights(
    horizon_weeks: int, half_life_weeks: float | None, discount_rate_annual: float
) -> np.ndarray:
    """Per-week retention × financial discount weights over the horizon."""
    w = np.arange(max(int(horizon_weeks), 1), dtype=float)
    ret = (
        np.power(0.5, w / max(float(half_life_weeks), _EPS))
        if half_life_weeks
        else np.ones_like(w)
    )
    disc = (
        np.power(1.0 + float(discount_rate_annual), -w / 52.0)
        if discount_rate_annual
        else np.ones_like(w)
    )
    return ret * disc


def compute_experiment_net_value(
    *,
    channel: str,
    evoi_kpi_units: float | None,
    evpi_kpi_units: float | None = None,
    kpi_delta_draws: np.ndarray | None = None,
    spend_delta: float = 0.0,
    margin_per_kpi: float | None = None,
    response_horizon_weeks: int = 26,
    half_life_weeks: float | None = None,
    discount_rate_annual: float = 0.0,
    model_anchored: bool = False,
    opportunity_cost_result: Any = None,
) -> ExperimentNetValue:
    """Net the reallocation gain against the test's short-term loss.

    Args:
        evoi_kpi_units: the experiment's EVOI (KPI-contribution units) — pass
            the realized-precision value from the anchor loopback when
            available and set ``model_anchored=True``.
        evpi_kpi_units: the EVPI ceiling; the gain is capped at it.
        kpi_delta_draws: per-draw signed window KPI delta of running the test
            (from ``compute_opportunity_cost(..., return_draws=True)``). Gives
            the net-value distribution + ``P(net > 0)``; omit for point-only.
        spend_delta: signed $ spend change over the window (negative = holdout
            savings).
        margin_per_kpi: $ per KPI unit (the opportunity-cost module's resolved
            value). ``None`` → KPI-units basis, spend excluded.
        half_life_weeks: information half-life (``eig.channel_half_life``);
            ``None`` disables the decay haircut.
        opportunity_cost_result: an ``OpportunityCostResult`` to pull draws /
            spend / margin / half-life defaults from (fields explicitly passed
            win).
    """
    warnings: list[str] = []

    # ── pull defaults off a provided OC result ──
    oc = opportunity_cost_result
    if oc is not None:
        if kpi_delta_draws is None:
            kpi_delta_draws = getattr(oc, "draws", None)
            if isinstance(kpi_delta_draws, dict):
                kpi_delta_draws = kpi_delta_draws.get("kpi_delta")
        if margin_per_kpi is None:
            margin_per_kpi = getattr(oc, "margin_per_kpi", None)
        if abs(spend_delta) < _EPS:
            spend_delta = float(getattr(oc, "spend_delta", 0.0) or 0.0)

    horizon = max(int(response_horizon_weeks), 1)
    dollar = margin_per_kpi is not None
    unit = "$" if dollar else "KPI units"
    m = float(margin_per_kpi) if dollar else 1.0

    # ── gain side ──
    basis = "insufficient"
    evoi_capped = None
    decay_factor = None
    gain = None
    if evoi_kpi_units is not None and math.isfinite(float(evoi_kpi_units)):
        evoi_capped = max(float(evoi_kpi_units), 0.0)
        if evpi_kpi_units is not None and math.isfinite(float(evpi_kpi_units)):
            evoi_capped = min(evoi_capped, max(float(evpi_kpi_units), 0.0))
        weights = _decay_weights(horizon, half_life_weeks, discount_rate_annual)
        decay_factor = float(weights.mean())
        gain = m * evoi_capped * decay_factor
        basis = "model_anchored" if model_anchored else "evoi_bounded"
        if not dollar:
            warnings.append(
                "No margin resolved — gain and loss are in KPI units and the "
                "spend change is excluded from the net."
            )
    else:
        warnings.append(
            "No EVOI available — run the priorities/anchor loopback first; "
            "only the loss side is reported."
        )

    # ── loss side (signed net profit of running the test) ──
    test_loss = test_loss_p5 = test_loss_p95 = None
    net_profit_mean = None
    net_draws = None
    if kpi_delta_draws is not None and np.asarray(kpi_delta_draws).size:
        d = np.asarray(kpi_delta_draws, dtype=float)
        d = d[np.isfinite(d)]
        if d.size:
            profit = m * d - (spend_delta if dollar else 0.0)  # (D,) signed
            net_profit_mean = float(profit.mean())
            loss = np.maximum(0.0, -profit)
            test_loss = float(loss.mean())
            test_loss_p5 = float(np.percentile(loss, 5))
            test_loss_p95 = float(np.percentile(loss, 95))
            if gain is not None:
                net_draws = gain + profit
    elif oc is not None:
        # summary-only fallback (no distribution)
        if dollar and getattr(oc, "net_profit_impact_median", None) is not None:
            net_profit_mean = float(oc.net_profit_impact_median)
            test_loss = float(oc.opportunity_cost_dollar_median or 0.0)
            test_loss_p95 = getattr(oc, "opportunity_cost_dollar_p95", None)
        elif not dollar:
            net_profit_mean = float(getattr(oc, "kpi_delta_median", 0.0) or 0.0)
            test_loss = float(getattr(oc, "forgone_kpi_median", 0.0) or 0.0)
            test_loss_p95 = getattr(oc, "forgone_kpi_p95", None)
        warnings.append(
            "Loss side from opportunity-cost summaries (no draws) — "
            "P(net > 0) unavailable; pass return_draws=True for the "
            "distribution."
        )

    # ── net ──
    net_value = net_p5 = net_p95 = prob_pos = None
    if gain is not None and net_profit_mean is not None:
        net_value = gain + net_profit_mean
        if net_draws is not None:
            net_p5 = float(np.percentile(net_draws, 5))
            net_p95 = float(np.percentile(net_draws, 95))
            prob_pos = float(np.mean(net_draws > 0.0))
    elif gain is not None:
        net_value = gain  # no measurable test cost supplied

    # ── break-even horizon ──
    breakeven = None
    if gain is not None and net_profit_mean is not None and evoi_capped is not None:
        realized_loss = max(0.0, -net_profit_mean)
        if realized_loss <= _EPS:
            breakeven = 0.0
        else:
            # weekly gain flow: horizon-total spread over H weeks with decay
            flow = (m * evoi_capped / horizon) * _decay_weights(
                horizon * 10, half_life_weeks, discount_rate_annual
            )
            cum = np.cumsum(flow)
            hit = np.nonzero(cum >= realized_loss)[0]
            breakeven = float(hit[0] + 1) if hit.size else None
            if breakeven is None:
                warnings.append(
                    "The decayed learning value never repays the test loss "
                    "within 10× the horizon — the test is hard to justify on "
                    "reallocation value alone."
                )

    return ExperimentNetValue(
        channel=channel,
        unit=unit,
        basis=basis,
        test_loss=test_loss,
        test_loss_p5=test_loss_p5,
        test_loss_p95=test_loss_p95,
        net_profit_during_test=net_profit_mean,
        evoi_raw=None if evoi_kpi_units is None else float(evoi_kpi_units),
        evpi_cap=None if evpi_kpi_units is None else float(evpi_kpi_units),
        decay_factor=decay_factor,
        reallocation_gain=gain,
        net_value=net_value,
        net_value_p5=net_p5,
        net_value_p95=net_p95,
        prob_net_positive=prob_pos,
        breakeven_horizon_weeks=breakeven,
        horizon_weeks=horizon,
        half_life_weeks=half_life_weeks,
        margin_per_kpi=margin_per_kpi,
        assumptions={
            "decay": "gain = EVOI × avg retention over horizon (half-life decay × discounting)",
            "loss_convention": "signed net profit; a money-saving holdout has negative loss",
            "evpi_capped": evpi_kpi_units is not None,
            "discount_rate_annual": discount_rate_annual,
            "spend_delta": float(spend_delta),
        },
        warnings=warnings,
    )


__all__ = ["ExperimentNetValue", "compute_experiment_net_value"]
