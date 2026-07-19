"""Suggest an experiment setup, and Pareto-optimize the design space.

Given a fitted MMM and a channel, this turns "test this channel" into a
recommended, runnable setup — design family, intensity, duration, test/control
groups (or a flighting schedule), and a **cool-down period** derived from the
channel's fitted adstock — by exploring a grid of candidate designs and ranking
them on FOUR objectives the client cares about:

- **lowest MDE** — the smallest ROAS effect the test can detect (precision),
- **highest statistical power** (as a shortfall below an 80% target) to detect
  the model's expected effect — for flighting designs the power is computed for
  the ROAS, contribution AND marginal-ROAS estimands separately (a multi-level
  spend schedule is what makes the marginal / saturation curve identifiable),
- **smallest short-term tradeoff** — the opportunity cost of deviating from
  business-as-usual (forgone KPI, or net-$ downside when a margin is known),
- **shortest duration** — weeks in market.

These trade off against each other (a longer, bigger test detects more but costs
more), so there is no single best design — there is a **Pareto front** of
non-dominated designs. We compute that front, then recommend the design at its
"knee" that is also powered to detect the model's expected effect.

Efficiency: the MDE for every duration comes from one pure-pandas power-curve
call per (footprint, intensity); the opportunity cost reuses a single shared BAU
posterior pass across the whole grid (only the perturbed pass is per-config).

numpy/pandas only — kernel-safe. Reuses ``planning.design`` (designs + power),
``planning.opportunity_cost`` (short-term risk), and the fitted adstock.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .design import flighting_design, geo_lift_design
from .opportunity_cost import compute_opportunity_cost

_EPS = 1e-9
# z_{0.975}; MDE_FACTOR = z_{0.975} + z_{0.80} = 2.8 (mirrors design.MDE_FACTOR).
_Z_975 = 1.959963984540054
_FACTOR = 2.8
DEFAULT_POWER_TARGET = 0.80


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard-normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2.0)))


def _power_from_se(se: float, effect_draws) -> float:
    """Signed two-sided power to detect the expected effect at standard error
    ``se``: ``mean_d[Φ(eff_d/se − z) + Φ(−eff_d/se − z)]`` over the effect
    posterior. A null effect scores ~alpha; a large effect → ~1."""
    draws = np.asarray(effect_draws, dtype=float)
    if draws.size == 0 or not math.isfinite(se) or se < 0:
        return float("nan")
    if se <= _EPS:
        return 1.0
    p = _phi(draws / se - _Z_975) + _phi(-draws / se - _Z_975)
    return float(np.clip(np.mean(p), 0.0, 1.0))


def _power_for(mde_roas: float, effect_draws) -> float:
    """Power of a design (with this MDE → SE = MDE/2.8) to detect the expected
    effect. At an effect equal to the MDE this returns ~0.80 by construction, so
    power and MDE stay consistent."""
    if not math.isfinite(mde_roas) or mde_roas <= 0:
        return float("nan")
    return _power_from_se(mde_roas / _FACTOR, effect_draws)


# Lower bound of the model's 95% posterior interval (equal-tailed) — the
# conservative "if the channel's true effect is at the pessimistic end" point.
_LOWER_Q = 2.5


def _power_at_effect(se: float, effect: float) -> float:
    """Signed two-sided power to detect a SINGLE effect value at standard error
    ``se`` — ``Φ(effect/se − z) + Φ(−effect/se − z)`` (the `_power_from_se`
    integrand evaluated at one effect rather than averaged over the posterior)."""
    if not (math.isfinite(se) and math.isfinite(effect)) or se < 0:
        return float("nan")
    if se <= _EPS:
        return 1.0
    z = _Z_975
    p = 0.5 * (1.0 + math.erf((effect / se - z) / math.sqrt(2.0))) + 0.5 * (
        1.0 + math.erf((-effect / se - z) / math.sqrt(2.0))
    )
    return float(min(1.0, max(0.0, p)))


def _lower_effect(effect_draws) -> float:
    """The lower 95% bound (``_LOWER_Q`` percentile) of an effect posterior."""
    draws = np.asarray(effect_draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan")
    return float(np.percentile(draws, _LOWER_Q))


def _power_lower(se: float, effect_draws) -> float:
    """Conservative power: power to detect the effect if it equals the model's
    lower 95% posterior bound (vs `_power_from_se`, which averages over draws)."""
    return _power_at_effect(se, _lower_effect(effect_draws))


def _national_curve_posteriors(
    mmm: Any, channel: str, *, max_draws: int, random_seed: int
) -> dict[str, np.ndarray] | None:
    """Per-draw posteriors of the channel's three flighting estimands from the
    model's response curve: total **contribution** (window-summed) at current
    spend, **average ROAS** (contribution / spend), and **marginal ROAS** (the
    curve slope at the operating point, by central difference)."""
    try:
        from .budget import compute_response_curves

        names = list(mmm.channel_names)
        if channel not in names:
            return None
        c = names.index(channel)
        curves = compute_response_curves(
            mmm,
            multipliers=(0.5, 1.0, 1.5),
            max_draws=max_draws,
            random_seed=random_seed,
        )
        mults = curves.multipliers
        i1 = int(np.argmin(np.abs(mults - 1.0)))
        ilo = int(np.argmin(np.abs(mults - 0.5)))
        ihi = int(np.argmin(np.abs(mults - 1.5)))
        spend = float(curves.base_spend[c])
        n_periods = max(int(getattr(mmm, "n_periods", 0) or 0), 1)
        if spend <= _EPS or ihi == ilo:
            return None
        total = curves.contributions[:, c, i1]  # (D,) window-total contribution
        # per-week g(x0): pairs with the per-week se_contribution so ROAS and
        # contribution detection power coincide (they are the same test).
        contribution = total / n_periods
        roas = total / spend  # average ROAS (= g(x0)/x0), per-week-invariant
        # marginal ROAS = curve slope per spend dollar at the operating point
        dspend = (mults[ihi] - mults[ilo]) * spend
        mroas = (
            curves.contributions[:, c, ihi] - curves.contributions[:, c, ilo]
        ) / dspend
        return {
            "contribution": contribution,
            "roas": roas,
            "mroas": mroas,
            # the MODEL's per-week operating spend — the SE basis the flighting
            # design must use so ROAS/contribution detection power coincide.
            "x0": spend / n_periods,
        }
    except Exception:  # noqa: BLE001 — flighting power just degrades to None
        return None


def _flighting_power_breakdown(
    estimand_ses: dict | None, posteriors: dict | None, *, target: float
) -> dict | None:
    """Power for the flighting design's three estimands separately — ROAS,
    contribution, mROAS — from the design SEs and the model posteriors. The
    binding ``min`` (ROAS vs mROAS; contribution ≡ ROAS detectability) drives the
    Pareto power objective so multi-level designs that actually pin the curve win.
    """
    if not estimand_ses or not posteriors:
        return None
    identified = bool(estimand_ses.get("mroas_identified"))
    p_roas = _power_from_se(
        estimand_ses.get("se_roas", float("nan")), posteriors["roas"]
    )
    p_contrib = _power_from_se(
        estimand_ses.get("se_contribution", float("nan")), posteriors["contribution"]
    )
    # The tangent mROAS is only the curve slope when the design identifies it
    # (≥3 levels). A binary on/off's "slope" is a secant — NOT the marginal — so
    # it must not earn or bind the curve-power objective.
    p_mroas = (
        _power_from_se(estimand_ses.get("se_mroas", float("nan")), posteriors["mroas"])
        if identified
        else float("nan")
    )
    # Conservative power at the model's LOWER 95% posterior bound for each
    # estimand (vs the expected/assurance power above, which averages over the
    # posterior) — "can the test still detect the channel if its true effect is
    # at the pessimistic end of what the model predicts?"
    lo_roas = _power_at_effect(
        estimand_ses.get("se_roas", float("nan")), _lower_effect(posteriors["roas"])
    )
    lo_contrib = _power_at_effect(
        estimand_ses.get("se_contribution", float("nan")),
        _lower_effect(posteriors["contribution"]),
    )
    lo_mroas = (
        _power_at_effect(
            estimand_ses.get("se_mroas", float("nan")),
            _lower_effect(posteriors["mroas"]),
        )
        if identified
        else float("nan")
    )
    # The binding power requires every estimand the design CLAIMS to measure
    # (ROAS always; the tangent mROAS when identified). If any required estimand
    # is non-finite (degenerate SE), the design's power is unknown, not the
    # surviving estimand — otherwise a design whose ROAS is unmeasurable could
    # look powered on mROAS alone.
    required = [p_roas] + ([p_mroas] if identified else [])
    binding = min(required) if all(math.isfinite(p) for p in required) else None
    required_lo = [lo_roas] + ([lo_mroas] if identified else [])
    binding_lo = (
        min(required_lo) if all(math.isfinite(p) for p in required_lo) else None
    )

    def _f(x):
        return x if math.isfinite(x) else None

    return {
        "roas": _f(p_roas),
        "contribution": _f(p_contrib),
        "mroas": _f(p_mroas),
        # power at the model's lower 95% contribution/ROAS/mROAS bound
        "roas_lower": _f(lo_roas),
        "contribution_lower": _f(lo_contrib),
        "mroas_lower": _f(lo_mroas),
        "lower_quantile": _LOWER_Q / 100.0,
        "mroas_identified": identified,
        "n_levels": int(estimand_ses.get("n_distinct_levels", 0)),
        "min": binding,
        "min_lower": binding_lo,
        "target": float(target),
    }


def _duration_grid(
    lo: int, hi: int, *, step: int = 4, max_n: int = 5
) -> tuple[int, ...]:
    """Integer durations spanning ``[lo, hi]`` weeks (endpoints included), spaced
    ~``step`` apart, capped at ``max_n`` points."""
    lo, hi = int(lo), int(hi)
    lo, hi = (hi, lo) if lo > hi else (lo, hi)
    lo = max(1, lo)
    if hi <= lo:
        return (lo,)
    n = int(np.clip(round((hi - lo) / step) + 1, 2, max_n))
    return tuple(sorted({int(round(v)) for v in np.linspace(lo, hi, n)}))


def _intensity_grid(
    lo: float, hi: float, *, step: float = 50.0, max_n: int = 4
) -> tuple[float, ...]:
    """Signed spend-variation %s spanning ``[lo, hi]`` (endpoints included),
    spaced ~``step`` apart, capped at ``max_n`` points. Floored at -100% (you
    cannot cut more than all of the spend)."""
    lo, hi = float(lo), float(hi)
    lo, hi = (hi, lo) if lo > hi else (lo, hi)
    lo = max(-100.0, lo)
    if abs(hi - lo) < _EPS:
        return (round(lo, 2),)
    n = int(np.clip(round((hi - lo) / step) + 1, 2, max_n))
    return tuple(dict.fromkeys(round(float(v), 2) for v in np.linspace(lo, hi, n)))


# ── Cool-down from adstock ─────────────────────────────────────────────────────


def cooldown_weeks(
    mmm: Any,
    channel: str,
    *,
    threshold: float = 0.05,
    min_weeks: int = 1,
    max_weeks: int = 26,
    default_weeks: int = 4,
) -> dict[str, Any]:
    """Weeks for the channel's carryover to wash out below ``threshold`` of the
    impulse, from the fitted geometric-adstock decay ``alpha``.

    After a holdout (or between flights) the channel's effect decays as
    ``alpha**k``; the cool-down is the smallest ``k`` with ``alpha**k <
    threshold`` — the time before the treated cells are back to true BAU and a
    post-test (or next-flight) reading is clean. It also sets the minimum
    flighting block length (a block shorter than the memory smears the contrast).
    """
    alpha = None
    half_life = None
    try:
        from mmm_framework.reporting.helpers import compute_adstock_weights

        res = compute_adstock_weights(mmm, channels=[channel]).get(channel)
        if res is not None:
            alpha = float(res.alpha_mean)
            half_life = float(res.half_life)
    except Exception:  # noqa: BLE001 — fall back to a conservative default
        alpha = None

    if alpha is not None and alpha >= 1.0:
        # non-decaying / explosive memory → longest washout (log would be <= 0).
        return {
            "cooldown_weeks": int(max_weeks),
            "alpha": alpha,
            "half_life": half_life,
            "basis": "non_decaying",
            "threshold": threshold,
        }
    if alpha is None or alpha <= 0.0:
        # adstock unknown → a moderate practical washout; ~no carryover → minimal.
        weeks = default_weeks if alpha is None else min_weeks
        return {
            "cooldown_weeks": int(np.clip(weeks, min_weeks, max_weeks)),
            "alpha": alpha,
            "half_life": half_life,
            "basis": "default" if alpha is None else "no_carryover",
            "threshold": threshold,
        }

    k = math.ceil(math.log(threshold) / math.log(alpha))
    return {
        "cooldown_weeks": int(np.clip(k, min_weeks, max_weeks)),
        "alpha": alpha,
        "half_life": half_life,
        "basis": "adstock_decay",
        "threshold": threshold,
    }


# ── Candidate evaluation ───────────────────────────────────────────────────────


@dataclass
class CandidateEval:
    """One evaluated experiment design with its three Pareto objectives + the
    runnable setup. JSON-safe via ``to_dict``."""

    index: int
    design_key: str
    mode: str  # holdout | scaling | flighting
    footprint: str  # full | half | national
    n_pairs: int | None
    intensity_pct: float
    duration: int
    # objectives (all "lower is better")
    mde_roas: float
    power_shortfall: float  # max(0, target - power); 0 once power >= target
    tradeoff: float  # net-$ downside when margin known, else forgone KPI
    tradeoff_basis: str  # net_dollar | forgone_kpi
    # supporting risk detail
    forgone_kpi_median: float
    opportunity_cost_dollar_median: float | None
    net_profit_impact_median: float | None
    spend_at_risk: float
    pct_of_window_kpi: float | None
    duration_effective: int
    # verdict
    powered: bool
    # statistical power to detect the model's expected effect (None if unknown)
    power: float | None = None
    # conservative power at the model's lower 95% effect bound (None if unknown)
    power_lower: float | None = None
    power_target: float = DEFAULT_POWER_TARGET
    # flighting only: power per estimand {roas, contribution, mroas, ...}
    power_breakdown: dict | None = None
    # net-value axis (filled by the post-pass when the EVOI anchor computes):
    # the design's precision, its surrogate EVOI (KPI units), the decayed/
    # EVPI-capped $ gain, and the netted decision figure gain − test loss.
    sigma_exp: float | None = None
    evoi_kpi: float | None = None
    reallocation_gain: float | None = None
    net_value: float | None = None
    net_value_basis: str | None = None
    on_pareto: bool = False
    is_recommended: bool = False
    # runnable setup
    treatment_geos: list[str] = field(default_factory=list)
    control_geos: list[str] = field(default_factory=list)
    schedule: list[dict] | None = None
    block_weeks: int | None = None
    duration_requested: int | None = None
    warnings: list[str] = field(default_factory=list)

    def objectives(self) -> tuple[float, float, float, float]:
        # lower is better on every axis: precision, power gap below target,
        # short-term cost, time in market.
        return (
            self.mde_roas,
            self.power_shortfall,
            self.tradeoff,
            float(self.duration),
        )

    def to_dict(self) -> dict[str, Any]:
        def _clean(v: Any) -> Any:
            if isinstance(v, float):
                return float(v) if math.isfinite(v) else None
            return v

        return {k: _clean(v) for k, v in asdict(self).items()}


def _tradeoff(oc, margin_known: bool) -> tuple[float, str]:
    """The short-term tradeoff objective.

    With a margin: the net-$ opportunity-cost downside (a money-saving holdout
    scores ~0 — a true signal). Without a margin: the forgone KPI for a holdout
    (lost sales), but a scaling-UP test GAINS KPI (forgone≈0), so its cost is the
    extra budget committed (``spend_at_risk``) — otherwise a +100% test would tie
    at 0 with a +50% test and dominate it for free. NB: without a margin the axis
    can mix lost-KPI (holdout) and $-committed (scaling) units across modes —
    ``tradeoff_basis`` records which, and the grid flags ``mixed_tradeoff_units``.
    """
    if margin_known and oc.opportunity_cost_dollar_median is not None:
        return float(oc.opportunity_cost_dollar_median), "net_dollar"
    forgone = float(oc.forgone_kpi_median)
    if forgone > _EPS:
        return forgone, "forgone_kpi"
    return float(oc.spend_at_risk), "spend_at_risk"


# ── Net-value axis (Gaussian EVOI surrogate) ──────────────────────────────────


def _evoi_anchor(
    mmm: Any,
    channel: str,
    effect_draws,
    sigma_lo: float,
    sigma_hi: float,
    *,
    max_draws: int,
    random_seed: int,
    n_steps: int = 400,
) -> dict[str, Any] | None:
    """TWO anchored preposterior-MC EVOIs (+ EVPI) for ``channel`` at the
    extremes of the grid's design precisions — the expensive half of the
    net-value axis, paid once per grid, then interpolated to every candidate
    by the fitted Gaussian surrogate (``evoi.fit_evoi_surrogate``; anchoring at
    the extremes keeps every candidate INSIDE the calibrated bracket, where the
    surrogate tracks the MC EVOI to ~±15%).

    ``effect_draws`` must be the incremental-ROAS posterior computed at the
    SAME ``max_draws`` as the curves (the draw-pairing convention the priority
    loopback uses); a length mismatch refuses rather than mispair."""
    try:
        from .budget import compute_response_curves, optimize_budget
        from .evoi import compute_evoi_for_channel, compute_evpi, fit_evoi_surrogate

        x = np.asarray(effect_draws, dtype=float)
        x = x[np.isfinite(x)]
        tau = float(x.std()) if x.size else 0.0
        if tau <= _EPS or not math.isfinite(sigma_lo) or sigma_lo <= 0:
            return None
        curves = compute_response_curves(
            mmm, max_draws=max_draws, random_seed=random_seed
        )
        if channel not in curves.channel_names:
            return None
        c = curves.channel_names.index(channel)
        if x.shape[0] != curves.contributions.shape[0]:
            return None  # not draw-paired with the curves — cannot reweight
        opt = optimize_budget(curves=curves, random_seed=random_seed)
        port = compute_evpi(
            curves,
            total_budget=opt.total_budget,
            per_draw_alloc=opt.per_draw_alloc,
            optimal_alloc=opt.optimal_alloc,
            n_steps=n_steps,
        )
        # Common random numbers across the two anchors so the fitted ratio
        # (which pins delta) isn't corrupted by independent MC noise.
        rng = np.random.default_rng(random_seed)
        D = curves.contributions.shape[0]
        outcome_draws = (rng.integers(0, D, size=48), rng.standard_normal(48))
        sigmas = [float(sigma_lo)]
        if math.isfinite(sigma_hi) and sigma_hi > sigma_lo * (1.0 + 1e-6):
            sigmas.append(float(sigma_hi))
        anchors: list[tuple[float, float]] = []
        for sig in sigmas:
            v = compute_evoi_for_channel(
                curves,
                c,
                x,
                sig,
                optimal_alloc=opt.optimal_alloc,
                total_budget=opt.total_budget,
                n_steps=n_steps,
                outcome_draws=outcome_draws,
            )
            if port.evpi > 0:
                v = min(v, port.evpi)
            anchors.append((sig, float(v)))
        surrogate = fit_evoi_surrogate(tau, anchors)
        if surrogate is None:
            return None
        return {
            "surrogate": surrogate,
            "anchors": anchors,
            "evpi": float(port.evpi),
            "tau": tau,
        }
    except Exception:  # noqa: BLE001 — the axis just degrades to the cost axis
        return None


def _apply_net_value_axis(
    cands: list[CandidateEval],
    anchor: dict[str, Any],
    *,
    channel: str,
    margin: float,
    response_horizon_weeks: int,
) -> bool:
    """Swap the tradeoff objective for ``−net_value`` (net value = decayed,
    EVPI-capped reallocation gain − signed test loss, in $) on EVERY candidate,
    pricing each design's EVOI with the fitted Gaussian surrogate at its own
    precision (``sigma_exp = MDE / 2.8``). All-or-nothing: if any candidate
    can't be priced, the axis is left untouched (a mixed axis would corrupt the
    front)."""
    from types import SimpleNamespace

    from .eig import channel_half_life
    from .experiment_value import compute_experiment_net_value

    half_life = channel_half_life(channel)
    surrogate = anchor["surrogate"]
    staged: list[tuple[CandidateEval, float, float, Any]] = []
    for c in cands:
        if not (math.isfinite(c.mde_roas) and c.mde_roas > 0):
            return False
        if c.net_profit_impact_median is None:
            return False
        se = c.mde_roas / _FACTOR
        evoi_c = surrogate(se, evpi=anchor["evpi"])
        oc_shim = SimpleNamespace(
            draws=None,
            margin_per_kpi=margin,
            spend_delta=0.0,
            net_profit_impact_median=c.net_profit_impact_median,
            opportunity_cost_dollar_median=c.opportunity_cost_dollar_median,
            opportunity_cost_dollar_p95=None,
        )
        nv = compute_experiment_net_value(
            channel=channel,
            evoi_kpi_units=evoi_c,
            evpi_kpi_units=anchor["evpi"],
            margin_per_kpi=margin,
            response_horizon_weeks=response_horizon_weeks,
            half_life_weeks=half_life,
            model_anchored=True,
            opportunity_cost_result=oc_shim,
        )
        if nv.net_value is None or not math.isfinite(nv.net_value):
            return False
        staged.append((c, se, evoi_c, nv))
    if not staged:
        return False
    for c, se, evoi_c, nv in staged:
        c.sigma_exp = float(se)
        c.evoi_kpi = float(evoi_c)
        c.reallocation_gain = nv.reallocation_gain
        c.net_value = float(nv.net_value)
        c.net_value_basis = nv.basis
        c.tradeoff = float(-nv.net_value)  # lower-better Pareto convention
        c.tradeoff_basis = "net_value"
    return True


def _mde_at(power_curve: list[dict], t: int) -> float:
    """MDE(ROAS) at duration ``t`` from a design's power curve — exact when ``t``
    is a curve point, else linearly interpolated (clamped to the curve range).
    Used to reconcile the MDE to a ragged panel's realized window."""
    pts = sorted(
        (
            (int(p["duration"]), float(p["mde_roas"]))
            for p in (power_curve or [])
            if p.get("mde_roas") is not None and math.isfinite(p["mde_roas"])
        ),
    )
    if not pts:
        return float("inf")
    for d, m in pts:
        if d == t:
            return m
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (d0, m0), (d1, m1) in zip(pts, pts[1:]):
        if d0 <= t <= d1 and d1 > d0:
            return m0 + (m1 - m0) * (t - d0) / (d1 - d0)
    return pts[-1][1]


def evaluate_experiment_grid(
    mmm: Any,
    dataset_path: str,
    kpi: str,
    channel: str,
    *,
    duration_min: int = 4,
    duration_max: int = 12,
    intensity_min: float = 50.0,
    intensity_max: float = 100.0,
    durations: tuple[int, ...] | None = None,
    scaling_intensities: tuple[float, ...] | None = None,
    include_holdout: bool = True,
    footprints: tuple[str, ...] = ("full", "half"),
    amplitudes: tuple[float, ...] | None = None,
    margin: float | None = None,
    price: float | None = None,
    kpi_kind: str = "revenue",
    power_target: float = DEFAULT_POWER_TARGET,
    net_value_axis: bool = True,
    response_horizon_weeks: int = 26,
    max_draws: int = 80,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a grid of candidate designs on FOUR objectives, all lower-better:
    MDE, power shortfall below ``power_target`` (default 80%), short-term cost,
    and duration.

    When a margin is known and ``net_value_axis`` (default), the cost objective
    is upgraded to the **net value of testing**: each candidate's EVOI (priced
    by the Gaussian surrogate at its own design precision, from ONE anchored
    preposterior-MC EVOI per grid) is decayed over ``response_horizon_weeks``,
    capped at EVPI, converted to $, and netted against the candidate's signed
    short-term test loss — the Pareto axis becomes ``−net_value`` and every
    candidate carries ``net_value``/``evoi_kpi``/``reallocation_gain``. If the
    anchor cannot compute (no draw pairing, degenerate posterior), the axis
    degrades to the existing net-$ downside.

    The design space is bounded by RANGES the caller controls: durations in
    ``[duration_min, duration_max]`` weeks and signed spend variations in
    ``[intensity_min, intensity_max]`` % (e.g. ``-100`` go dark … ``+150`` scale
    up; intensities at -100% render as a go-dark holdout). The optimizer
    auto-samples a few points within each range; pass explicit ``durations`` /
    ``scaling_intensities`` to override the sampling.

    Reuses one pure-pandas power-curve call per (footprint, intensity) for the
    MDE of every duration, and one shared BAU posterior pass for the whole grid's
    opportunity cost. Power is the model-anchored probability each design detects
    the expected effect (no extra posterior passes).
    """
    from .design import design_options

    if durations is None:
        durations = _duration_grid(duration_min, duration_max)
    if scaling_intensities is None:
        scaling_intensities = _intensity_grid(intensity_min, intensity_max)
    if amplitudes is None:
        # National flighting amplitude (±%, in (0, 100]) from the spend range.
        amps = sorted(
            {min(100.0, abs(x)) for x in scaling_intensities if abs(x) > _EPS}
        )
        amplitudes = tuple(amps) or (40.0,)

    opts = design_options(dataset_path, kpi, channel)
    geo = "geo_lift" in opts["designs"]
    n_geos = int(opts.get("n_geos", 0) or 0)
    margin_known = margin is not None and float(margin) > 0

    cool = cooldown_weeks(mmm, channel)
    # Flighting blocks must be >= the adstock memory; use the half-life.
    suggested_block = max(2, int(math.ceil((cool.get("half_life") or 2) or 2)))

    # One shared BAU pass for the whole grid.
    x_bau = np.asarray(getattr(mmm, "X_media_raw"), dtype=np.float64)
    contrib_bau = mmm.sample_channel_contributions(
        X_media=x_bau, max_draws=max_draws, random_seed=random_seed
    )

    # Reference expected effect: the model's incremental-ROAS posterior that each
    # design's statistical power is measured against.
    expected_roas, expected_kpi, effect_draws = _reference_effect(
        mmm,
        dataset_path,
        kpi,
        channel,
        geo,
        n_geos,
        contrib_bau,
        max_draws,
        random_seed,
    )

    cands: list[CandidateEval] = []
    notes: list[str] = []
    idx = 0
    if geo:
        max_pairs = max(1, n_geos // 2)
        fp_pairs = {"full": None, "half": max(1, max_pairs // 2)}
        # Spend-variation grid → (mode, signed intensity). A variation at -100%
        # IS a go-dark holdout; dedup it against the explicit holdout baseline.
        variants: list[tuple[str, float]] = []
        if include_holdout:
            variants.append(("holdout", -100.0))
        for x in scaling_intensities:
            mode_x = "holdout" if x <= -99.5 else "scaling"
            if mode_x == "holdout" and include_holdout:
                continue
            variants.append((mode_x, float(x)))

        for fp_name in footprints:
            if fp_name not in fp_pairs:
                continue
            n_pairs = fp_pairs[fp_name]
            for mode, intensity in variants:
                try:
                    base = geo_lift_design(
                        dataset_path,
                        kpi,
                        channel,
                        design=mode,
                        # signed for scaling (negative = reduction); ignored for holdout.
                        intensity_pct=float(intensity) if mode == "scaling" else 50.0,
                        n_pairs=n_pairs,
                        durations=tuple(durations),
                        randomize=True,
                        seed=random_seed,
                    )
                except ValueError:
                    continue
                power_curve = base.get("power_curve") or []
                for d in durations:
                    cfg = dict(base)
                    cfg["duration"] = int(d)
                    ev = _eval_one(
                        mmm,
                        cfg,
                        idx,
                        mode,
                        fp_name,
                        base.get("n_pairs"),
                        float(base["intensity_pct"]),
                        int(d),
                        power_curve,
                        margin,
                        price,
                        kpi_kind,
                        margin_known,
                        effect_draws,
                        power_target,
                        contrib_bau,
                        max_draws,
                        random_seed,
                    )
                    if ev is not None:
                        ev.treatment_geos = list(base.get("treatment_geos") or [])
                        ev.control_geos = list(base.get("control_geos") or [])
                        cands.append(ev)
                        idx += 1
    else:
        # National flighting. The spend-variation range becomes the schedule's
        # spend LEVELS (as multipliers of mean spend): a 2-level on/off pins the
        # average ROAS, and a >=3-level schedule additionally traces the response
        # CURVE so the marginal ROAS / saturation is identified. Each design's
        # power is reported for ROAS, contribution AND mROAS separately, and the
        # BINDING one drives the Pareto power objective. (A block shorter than the
        # adstock memory smears the contrast, so the schedule floors at
        # block_weeks*2 — short requested durations can collapse; de-dup + note.)
        posteriors = _national_curve_posteriors(
            mmm, channel, max_draws=max_draws, random_seed=random_seed
        )
        mult_grid = sorted(
            {round(max(0.0, 1.0 + x / 100.0), 6) for x in scaling_intensities}
        )
        level_sets: list[tuple[str, tuple | None]] = []
        if len(mult_grid) >= 2:
            level_sets.append(("on/off", (mult_grid[0], mult_grid[-1])))
        if len(mult_grid) >= 3:
            level_sets.append(("multi-level", tuple(mult_grid)))
        if not level_sets:
            level_sets.append(("on/off", None))  # amplitude-based fallback

        # The model's per-week operating spend — passed to flighting_design so its
        # estimand SEs and the curve posteriors share one spend basis (then ROAS
        # and contribution detection power coincide regardless of CSV/model window).
        x0_model = posteriors.get("x0") if posteriors else None
        requested = sorted({int(d) for d in durations})
        seen_eff: set[tuple] = set()
        for _label, levels in level_sets:
            for d in durations:
                try:
                    base = flighting_design(
                        dataset_path,
                        kpi,
                        channel,
                        amplitude_pct=float(amplitudes[0]) if amplitudes else 40.0,
                        block_weeks=suggested_block,
                        duration=int(d),
                        levels=levels,
                        x0=float(x0_model) if x0_model else None,
                        seed=random_seed,
                    )
                except ValueError:
                    continue
                if not (base.get("weekly_spend_delta") or 0) > 0:
                    continue  # no spend to flight — drop (mirrors the geo guard)
                eff_dur = int(base.get("duration", d))
                n_lv = int(base.get("n_levels", 2))
                key = (n_lv, eff_dur)
                if key in seen_eff:
                    continue  # collapsed onto an already-evaluated (levels, window)
                seen_eff.add(key)
                breakdown = _flighting_power_breakdown(
                    base.get("estimand_ses"), posteriors, target=power_target
                )
                # MDE axis from the design-matrix avg-ROAS SE (consistent with the
                # ROAS power), falling back to the on/off contrast SE.
                es = base.get("estimand_ses") or {}
                mde = _FACTOR * es["se_roas"] if es.get("se_roas") else base["mde_roas"]
                top_level = max(base.get("levels") or [1.0])
                ev = _eval_one(
                    mmm,
                    base,
                    idx,
                    "flighting",
                    "national",
                    None,
                    float((top_level - 1.0) * 100.0),  # peak spend variation %
                    eff_dur,
                    [{"duration": eff_dur, "mde_roas": mde}],
                    margin,
                    price,
                    kpi_kind,
                    margin_known,
                    effect_draws,
                    power_target,
                    contrib_bau,
                    max_draws,
                    random_seed,
                    flighting_breakdown=breakdown,
                )
                if ev is not None:
                    ev.schedule = base.get("schedule")
                    ev.block_weeks = suggested_block
                    cands.append(ev)
                    idx += 1
        realized = {k[1] for k in seen_eff}
        if realized and len(realized) < len(requested):
            notes.append(
                f"Requested durations {requested} collapsed to {sorted(realized)} "
                f"weeks: the channel's adstock memory (block {suggested_block}w) "
                f"forces a minimum flight of {suggested_block * 2}w, so shorter "
                "tests cannot form a valid high/low contrast."
            )

    # ── Net-value axis: swap the cost objective for −(gain − loss) when a
    # margin is known and the one-shot EVOI anchor computes. Done BEFORE the
    # Pareto pass so the front and the knee rank on net value.
    net_axis_applied = False
    anchor: dict[str, float] | None = None
    if net_value_axis and margin_known and cands and len(effect_draws) > 0:
        ses = [
            c.mde_roas / _FACTOR
            for c in cands
            if math.isfinite(c.mde_roas) and c.mde_roas > 0
        ]
        if ses and len(ses) == len(cands):
            anchor = _evoi_anchor(
                mmm,
                channel,
                effect_draws,
                float(min(ses)),
                float(max(ses)),
                max_draws=max_draws,
                random_seed=random_seed,
            )
            if anchor is not None:
                net_axis_applied = _apply_net_value_axis(
                    cands,
                    anchor,
                    channel=channel,
                    margin=float(margin),
                    response_horizon_weeks=int(response_horizon_weeks),
                )
    if net_axis_applied:
        notes.append(
            "Cost objective upgraded to NET VALUE: decayed, EVPI-capped "
            "reallocation gain (Gaussian-surrogate EVOI at each design's own "
            "precision) minus the signed short-term test loss."
        )

    front = pareto_front(cands)
    for i in front:
        cands[i].on_pareto = True
    rec = recommend(cands, front)
    if rec is not None:
        cands[rec].is_recommended = True

    # Tradeoff-axis unit: a single comparable scale per result, except the
    # no-margin case can mix forgone-KPI (holdout) with $-committed (scaling).
    bases = {c.tradeoff_basis for c in cands}
    if bases == {"net_value"}:
        tradeoff_label = "net value of testing ($, higher is better)"
    elif bases == {"net_dollar"}:
        tradeoff_label = "short-term cost ($)"
    elif bases <= {"net_dollar", "spend_at_risk"}:
        tradeoff_label = "budget at risk ($)"
    elif bases == {"forgone_kpi"}:
        tradeoff_label = "forgone KPI"
    else:
        tradeoff_label = "short-term cost (mixed units)"
    mixed = len([b for b in bases if b != "net_dollar"]) > 1
    if mixed:
        notes.append(
            "Tradeoff axis mixes lost-KPI (holdout) and $-committed (scaling) "
            "units — supply a margin for a unified net-$ cost comparison."
        )

    return {
        "channel": channel,
        "kpi": kpi,
        "kind": "geo" if geo else "national",
        "cooldown": cool,
        "suggested_block_weeks": suggested_block,
        "expected_incremental_roas": expected_roas,
        "expected_incremental_kpi": expected_kpi,
        "margin_known": margin_known,
        "tradeoff_label": tradeoff_label,
        "mixed_tradeoff_units": mixed,
        "net_value_axis": net_axis_applied,
        "response_horizon_weeks": int(response_horizon_weeks),
        # JSON-safe provenance of the fitted surrogate (the callable stays out)
        "evoi_anchor": (
            {
                "anchors": [list(a) for a in anchor["anchors"]],
                "evpi": anchor["evpi"],
                "tau": anchor["tau"],
                "k": float(anchor["surrogate"].k),
                "delta": float(anchor["surrogate"].delta),
            }
            if net_axis_applied and anchor is not None
            else None
        ),
        "power_target": float(power_target),
        "design_space": {
            "duration_min": int(min(durations)) if durations else duration_min,
            "duration_max": int(max(durations)) if durations else duration_max,
            "durations": [int(d) for d in durations],
            "intensity_min": (
                float(min(scaling_intensities))
                if scaling_intensities
                else intensity_min
            ),
            "intensity_max": (
                float(max(scaling_intensities))
                if scaling_intensities
                else intensity_max
            ),
            "scaling_intensities": [float(x) for x in scaling_intensities],
            "include_holdout": bool(include_holdout),
        },
        "n_candidates": len(cands),
        "pareto_indices": front,
        "recommended_index": rec,
        "notes": notes,
        "candidates": [c.to_dict() for c in cands],
    }


def _reference_effect(
    mmm, dataset_path, kpi, channel, geo, n_geos, contrib_bau, max_draws, random_seed
) -> tuple[float | None, float | None, list]:
    """The model's expected incremental-ROAS posterior for a representative
    design — the effect each candidate's power is measured against. Returns
    ``(median, expected_kpi_median, roas_draws)``."""
    try:
        from .design_anchor import model_anchored_effect

        if geo:
            ref = geo_lift_design(
                dataset_path,
                kpi,
                channel,
                design="scaling",
                intensity_pct=50.0,
                duration=8,
                randomize=True,
                seed=random_seed,
            )
        else:
            ref = flighting_design(
                dataset_path,
                kpi,
                channel,
                amplitude_pct=40.0,
                duration=12,
                seed=random_seed,
            )
        anchor = model_anchored_effect(
            mmm,
            ref,
            max_draws=max_draws,
            random_seed=random_seed,
            contrib_bau=contrib_bau,
        )
        return (
            float(anchor["incremental_roas_median"]),
            float(anchor["expected_incremental_kpi_median"]),
            list(anchor.get("incremental_roas_draws") or []),
        )
    except Exception:  # noqa: BLE001 — power just degrades to None (unknown)
        return None, None, []


def _eval_one(
    mmm,
    cfg,
    idx,
    mode,
    footprint,
    n_pairs,
    intensity,
    duration,
    power_curve,
    margin,
    price,
    kpi_kind,
    margin_known,
    effect_draws,
    power_target,
    contrib_bau,
    max_draws,
    random_seed,
    flighting_breakdown=None,
) -> CandidateEval | None:
    try:
        oc = compute_opportunity_cost(
            mmm,
            cfg,
            margin_per_kpi=margin,
            price=price,
            kpi_kind=kpi_kind,
            contrib_bau=contrib_bau,
            max_draws=max_draws,
            random_seed=random_seed,
        )
    except Exception:  # noqa: BLE001 — a bad config is just dropped from the grid
        return None
    tradeoff, basis = _tradeoff(oc, margin_known)
    # On a ragged panel the opportunity cost windows on a SHORTER realized window
    # than requested. Reconcile all the objectives onto that realized window so
    # the MDE (read at the realized duration) and the duration axis describe the
    # same window the cost does — otherwise the MDE would be optimistically read
    # at the longer requested duration.
    eff_dur = int(oc.duration_effective) if oc.duration_effective else int(duration)
    eff_dur = min(eff_dur, int(duration))
    mde_roas = _mde_at(power_curve, eff_dur)

    # Statistical power to detect the model's expected effect, and the shortfall
    # below the target (the 4th Pareto objective — 0 once power >= target).
    # Flighting designs measure THREE estimands (ROAS, contribution, mROAS) — the
    # binding power (min) drives the objective so a design must pin the curve, not
    # just the average. Geo designs use the single incremental-ROAS power.
    is_flighting = cfg.get("design_key") == "national_flighting"
    power_breakdown = None
    if flighting_breakdown and flighting_breakdown.get("min") is not None:
        power = float(flighting_breakdown["min"])
        power_breakdown = flighting_breakdown
    elif is_flighting:
        # Flighting power is the average-ROAS/curve breakdown only — never the geo
        # incremental-ROAS proxy (a different estimand). Unknown if unavailable.
        power = float("nan")
        power_breakdown = flighting_breakdown  # per-estimand values (min may be None)
    else:
        power = _power_for(mde_roas, effect_draws) if effect_draws else float("nan")
    if math.isfinite(power):
        power_shortfall = max(0.0, float(power_target) - power)
        powered = power >= float(power_target)
        power_val: float | None = float(power)
    else:
        # power unknown (no model effect) → neutral on the power axis; degrade to
        # the precision/cost/duration trade rather than penalizing everything.
        power_shortfall = 0.0
        powered = False
        power_val = None

    # Conservative power at the model's lower 95% effect bound: the flighting
    # binding-lower for a curve design, else the geo incremental-ROAS lower power.
    if power_breakdown is not None:
        _pl = power_breakdown.get("min_lower")
        power_lower = float(_pl) if isinstance(_pl, (int, float)) else None
    elif effect_draws is not None and math.isfinite(mde_roas) and mde_roas > 0:
        _pl = _power_lower(mde_roas / _FACTOR, effect_draws)
        power_lower = float(_pl) if math.isfinite(_pl) else None
    else:
        power_lower = None

    return CandidateEval(
        index=idx,
        design_key=cfg.get("design_key", ""),
        mode=mode,
        footprint=footprint,
        n_pairs=int(n_pairs) if n_pairs is not None else None,
        intensity_pct=float(intensity),
        duration=int(eff_dur),
        mde_roas=float(mde_roas),
        power_shortfall=float(power_shortfall),
        tradeoff=float(tradeoff),
        tradeoff_basis=basis,
        power=power_val,
        power_lower=power_lower,
        power_target=float(power_target),
        power_breakdown=power_breakdown,
        forgone_kpi_median=float(oc.forgone_kpi_median),
        opportunity_cost_dollar_median=oc.opportunity_cost_dollar_median,
        net_profit_impact_median=oc.net_profit_impact_median,
        spend_at_risk=float(oc.spend_at_risk),
        pct_of_window_kpi=oc.pct_of_window_kpi,
        duration_effective=int(oc.duration_effective),
        powered=powered,
        duration_requested=int(duration),
        warnings=list(oc.warnings or []),
    )


# ── Pareto front ───────────────────────────────────────────────────────────────


def _dominates(a: CandidateEval, b: CandidateEval) -> bool:
    """a dominates b iff a is <= b on every objective and < on at least one."""
    oa, ob = a.objectives(), b.objectives()
    return all(x <= y + _EPS for x, y in zip(oa, ob)) and any(
        x < y - _EPS for x, y in zip(oa, ob)
    )


def pareto_front(cands: list[CandidateEval]) -> list[int]:
    """Indices of the non-dominated designs over (MDE, power shortfall, cost,
    duration).
    Designs with a non-finite objective can never be on the front."""
    feasible = [c for c in cands if all(math.isfinite(o) for o in c.objectives())]
    front: list[int] = []
    for c in feasible:
        if not any(_dominates(o, c) for o in feasible if o is not c):
            front.append(c.index)
    return sorted(front)


def recommend(cands: list[CandidateEval], front: list[int]) -> int | None:
    """The recommended design: among the powered designs on the front (or all
    front designs if none are powered), the 'knee' — the one closest to the
    ideal point in objectives normalized over the front."""
    if not front:
        return None
    by_index = {c.index: c for c in cands}
    on_front = [by_index[i] for i in front if i in by_index]
    if not on_front:
        return None
    powered = [c for c in on_front if c.powered]
    pool = powered or on_front

    objs = np.array([c.objectives() for c in on_front], dtype=float)
    lo = objs.min(axis=0)
    span = np.where(objs.max(axis=0) - lo > _EPS, objs.max(axis=0) - lo, 1.0)
    best_i, best_d = None, float("inf")
    for c in pool:
        z = (np.array(c.objectives(), dtype=float) - lo) / span
        d = float(np.sqrt((z**2).sum()))
        if d < best_d:
            best_d, best_i = d, c.index
    return best_i


# ── Orchestration ──────────────────────────────────────────────────────────────


def suggest_experiment(
    mmm: Any,
    dataset_path: str,
    kpi: str,
    channel: str,
    *,
    margin: float | None = None,
    price: float | None = None,
    kpi_kind: str = "revenue",
    duration_min: int = 4,
    duration_max: int = 12,
    intensity_min: float = 50.0,
    intensity_max: float = 100.0,
    durations: tuple[int, ...] | None = None,
    scaling_intensities: tuple[float, ...] | None = None,
    include_holdout: bool = True,
    footprints: tuple[str, ...] = ("full", "half"),
    power_target: float = DEFAULT_POWER_TARGET,
    net_value_axis: bool = True,
    response_horizon_weeks: int = 26,
    max_draws: int = 80,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Recommend a runnable experiment setup for ``channel`` and return the full
    Pareto front (MDE × power shortfall below ``power_target`` × short-term cost
    × duration), over a design space bounded by the duration and spend-variation
    ranges. The recommended design carries its test/control groups (or
    flighting schedule), duration, intensity, and the adstock-derived cool-down.

    With a known margin the cost axis is the **net value of testing** (see
    :func:`evaluate_experiment_grid`) — gain from the learning minus the
    short-term loss — so the front trades precision and power against the
    dollars the test actually creates or destroys.
    """
    grid = evaluate_experiment_grid(
        mmm,
        dataset_path,
        kpi,
        channel,
        duration_min=duration_min,
        duration_max=duration_max,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        durations=durations,
        scaling_intensities=scaling_intensities,
        include_holdout=include_holdout,
        footprints=footprints,
        margin=margin,
        price=price,
        kpi_kind=kpi_kind,
        power_target=power_target,
        net_value_axis=net_value_axis,
        response_horizon_weeks=response_horizon_weeks,
        max_draws=max_draws,
        random_seed=random_seed,
    )
    cands = grid["candidates"]
    rec_i = grid["recommended_index"]
    recommended = cands[rec_i] if rec_i is not None else None
    grid["recommended"] = recommended
    grid["pareto"] = [cands[i] for i in grid["pareto_indices"]]
    return grid
