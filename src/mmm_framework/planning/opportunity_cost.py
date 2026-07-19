"""Short-term risk / opportunity cost of running an experiment, from the model.

A design perturbs the tested channel's spend away from business-as-usual (BAU)
over the test window — a holdout goes dark (-100%) in the treated cells, a
scaling test lifts spend by +X%, a national flighting schedule pulses on/off.
That perturbation changes the KPI during the test, and clients want to know the
**short-term cost** of buying the measurement: how much KPI (and money) do we
forgo by deviating from BAU?

We answer it from the fitted model, with posterior uncertainty:

1. Build ``X_experiment`` — a copy of the training media matrix with ONLY the
   treated geo × test-window rows perturbed (the rest is byte-identical BAU).
2. Read the counterfactual per-channel contributions for BOTH the BAU and the
   experiment media via ``mmm.sample_channel_contributions``. That node is a
   PyMC *Deterministic* (no observation noise), and identical ``max_draws``
   gives identical posterior thinning, so the two passes are **draw-paired
   exactly** — the per-draw KPI delta is exact, not a noisy difference (F1).
3. Summarize the per-draw KPI delta into expected / worst-case loss, the
   probability of a loss, the deterministic spend delta, an optional net-$
   business impact (when a margin/price is supplied), and a learning-vs-cost
   ratio against the experiment's EVOI.

Sign conventions are load-bearing and unit-tested:

- ``kpi_delta`` is SIGNED (negative = lost KPI). ``forgone_kpi`` is the
  non-negative downside ``max(0, -kpi_delta)``.
- ``spend_delta`` is SIGNED and computed INTERNALLY from the perturbed vs BAU
  matrices (negative for a holdout = money saved). We NEVER read
  ``design['weekly_spend_delta']`` — it is an absolute magnitude and would
  invert the holdout's net (F6).
- ``net_profit_impact = margin * kpi_delta - spend_delta`` is SIGNED;
  ``opportunity_cost_dollar = max(0, -net_profit_impact)`` is the non-negative
  cost the client asked for.

numpy/pandas only — kernel-safe (runs inside the sandboxed session kernels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# z_{0.975} + z_{0.80}; mirrors design.MDE_FACTOR.
_EPS = 1e-9


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class OpportunityCostResult:
    """Per-draw short-term risk of a design, summarized. JSON-safe via to_dict."""

    channel: str
    kpi: str
    design_key: str
    design_type: str
    # window / footprint (effective, after ragged-coverage intersection)
    duration_requested: int
    duration_effective: int
    n_treated_cells: int
    n_test_rows: int
    n_draws: int
    carryover_basis: str
    # KPI-unit risk (always present)
    expected_kpi_delta: float
    kpi_delta_median: float
    kpi_delta_p5: float
    kpi_delta_p95: float
    kpi_delta_with_carryover_median: float
    forgone_kpi_median: float
    forgone_kpi_p95: float
    prob_kpi_loss: float
    pct_of_window_kpi: float | None
    # spend (deterministic, signed)
    spend_delta: float
    abs_spend_change: float
    spend_at_risk: float
    # net-$ risk (None unless a margin resolves)
    margin_per_kpi: float | None
    margin_source: str
    kpi_kind: str
    net_profit_impact_median: float | None
    net_profit_impact_p5: float | None
    net_profit_impact_p95: float | None
    opportunity_cost_dollar_median: float | None
    opportunity_cost_dollar_p95: float | None
    prob_net_loss: float | None
    prob_loss_over_threshold: float | None
    loss_threshold: float | None
    # learning-vs-cost
    evoi_kpi_units: float | None
    evoi_per_week: float | None
    cost_per_week: float | None
    learning_to_cost_ratio: float | None
    learning_to_cost_basis: str
    response_horizon_weeks: int | None
    # status / honesty
    low_information: bool
    extrapolation_warning: bool
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # per-draw arrays (opt-in via return_draws=True; NOT serialized) — the
    # net-value module reuses these for its distribution, no extra passes.
    draws: dict | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        def _clean(v: Any) -> Any:
            if isinstance(v, float):
                return float(v) if np.isfinite(v) else None
            return v

        return {k: _clean(v) for k, v in self.__dict__.items() if k != "draws"}


# ── Geo / window resolution (F2, F3, F6) ──────────────────────────────────────

_GEO_DESIGNS = {"geo_lift", "matched_market_did"}


def _geo_names(mmm: Any) -> list[str]:
    return list(getattr(mmm, "geo_names", []) or [])


def _resolve_treated_geo_codes(mmm: Any, design: dict) -> tuple[list[int], list[str]]:
    """Geo codes (into ``mmm.geo_idx``) the design treats, plus warnings.

    Geo designs resolve ``design['treatment_geos']`` (names from a raw-csv pivot
    — a DIFFERENT pipeline than ``mmm.geo_names``) by stripped, then
    case-insensitive, name match; unmatched names raise so an empty mask can
    never be reported as "zero risk" (F6). National designs treat every geo.
    """
    design_key = design.get("design_key", "")
    has_geo = bool(getattr(mmm, "has_geo", False))
    n_geos = int(getattr(mmm, "n_geos", 1) or 1)

    if design_key in _GEO_DESIGNS:
        if not has_geo:
            raise ValueError(
                f"Design '{design_key}' is geo-based but the fitted model has no "
                "geography dimension — refit on a geo panel or use national flighting."
            )
        names = _geo_names(mmm)
        by_strip = {str(g).strip(): j for j, g in enumerate(names)}
        by_lower = {str(g).strip().lower(): j for j, g in enumerate(names)}
        codes: list[int] = []
        missing: list[str] = []
        for g in design.get("treatment_geos") or []:
            key = str(g).strip()
            if key in by_strip:
                codes.append(by_strip[key])
            elif key.lower() in by_lower:
                codes.append(by_lower[key.lower()])
            else:
                missing.append(str(g))
        if missing:
            raise ValueError(
                "Treatment geos not found in the fitted model's geographies "
                f"({', '.join(map(str, names))}): {', '.join(missing)}. The design "
                "and the model may have been built from different data."
            )
        if not codes:
            raise ValueError("Design carries no treatment geos to perturb.")
        return sorted(set(codes)), []

    # National / flighting: every geo is "treated" by the schedule.
    return list(range(max(n_geos, 1))), []


def _resolve_treated_rows(
    mmm: Any, design: dict, *, duration: int
) -> tuple[np.ndarray, list[int], np.ndarray, int, list[str]]:
    """(treated_mask, treated_geo_codes, window_codes, duration_effective, warnings).

    For GEO designs (treatment vs control) the window is the per-geo
    INTERSECTION of available period codes (the arms must compare an aligned
    window; the panel is ragged on the agent path — F3). National flighting has
    NO control arm — it applies a per-period multiplier and never compares geos
    — so it uses the UNION of coverage, landing the pulse on the actual most
    recent weeks. Either way the window is the last ``duration`` codes, and
    ``treated_mask`` selects treated-geo × window rows.
    """
    geo_idx = np.asarray(getattr(mmm, "geo_idx"), dtype=np.int64)
    time_idx = np.asarray(getattr(mmm, "time_idx"), dtype=np.int64)
    treated_geo_codes, warnings = _resolve_treated_geo_codes(mmm, design)
    is_geo_design = design.get("design_key", "") in _GEO_DESIGNS

    if is_geo_design:
        # Treatment and control must share an aligned window → intersection.
        coverage: set[int] | None = None
        for g in treated_geo_codes:
            codes_g = set(time_idx[geo_idx == g].tolist())
            coverage = codes_g if coverage is None else (coverage & codes_g)
        period_codes = coverage or set()
        if not period_codes:
            raise ValueError(
                "Treated geos share no common reporting periods — cannot form a "
                "test window from this panel."
            )
    else:
        # National pulse: no control arm, alignment across geos not required.
        period_codes = set()
        for g in treated_geo_codes:
            period_codes |= set(time_idx[geo_idx == g].tolist())
        if not period_codes:
            raise ValueError("Treated geos report no periods — cannot form a window.")

    sorted_common = sorted(period_codes)
    window_codes = np.array(sorted_common[-int(max(duration, 1)) :], dtype=np.int64)
    duration_effective = int(window_codes.size)
    if duration_effective < int(duration):
        warnings.append(
            f"Test window shrank to {duration_effective} of {int(duration)} weeks: "
            "the treated geos do not all report the most recent weeks (ragged panel)."
        )

    treated_mask = np.isin(geo_idx, treated_geo_codes) & np.isin(time_idx, window_codes)
    if not treated_mask.any():
        raise ValueError(
            "No model rows matched the treated geos × test window — the design and "
            "the fitted model appear to come from different data."
        )
    return treated_mask, treated_geo_codes, window_codes, duration_effective, warnings


def build_experiment_media(
    mmm: Any,
    design: dict,
    *,
    treated_mask: np.ndarray,
    window_codes: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """(X_experiment, channel_idx, n_test_rows).

    A float64 copy of ``mmm.X_media_raw`` with the tested channel column scaled
    on ``treated_mask`` rows: holdout → ``max(0, 1+intensity/100)`` (=0 at -100),
    scaling → ``1+intensity/100``, national_flighting → the per-period multiplier
    from ``design['schedule']`` mapped through the window's period codes. BAU
    everywhere else.
    """
    channel = design["channel"]
    names = list(getattr(mmm, "channel_names"))
    if channel not in names:
        raise ValueError(
            f"Channel '{channel}' is not in the fitted model "
            f"({', '.join(map(str, names))})."
        )
    ch_idx = names.index(channel)
    x_exp = np.asarray(getattr(mmm, "X_media_raw"), dtype=np.float64).copy()
    n_test_rows = int(treated_mask.sum())

    design_key = design.get("design_key", "")
    if design_key == "national_flighting":
        schedule = design.get("schedule") or []
        # week_offset -> the w-th period code of the (sorted) window.
        mult_by_code: dict[int, float] = {}
        for s in schedule:
            w = int(s.get("week_offset", 0))
            if 0 <= w < window_codes.size:
                mult_by_code[int(window_codes[w])] = float(s.get("multiplier", 1.0))
        time_idx = np.asarray(getattr(mmm, "time_idx"), dtype=np.int64)
        rows = np.where(treated_mask)[0]
        for r in rows:
            m = mult_by_code.get(int(time_idx[r]), 1.0)
            x_exp[r, ch_idx] *= m
    else:
        intensity = float(design.get("intensity_pct", 0.0))
        mult = max(0.0, 1.0 + intensity / 100.0)
        x_exp[treated_mask, ch_idx] *= mult

    return x_exp, ch_idx, n_test_rows


# ── Margin resolution ─────────────────────────────────────────────────────────


def _resolve_margin(
    preferences: dict | None,
    branding: dict | None,
    margin_per_kpi: float | None,
    kpi_kind: str,
    price: float | None,
) -> tuple[float | None, str]:
    """(value-per-KPI-unit, source). Order: positive param > branding economics
    > preferences['economics'] > (None, 'none'). ``kpi_kind='units'`` needs a
    price (value = margin*price); 'revenue' uses margin directly; 'other' → no $.
    """
    kind = (kpi_kind or "revenue").lower()
    if kind == "other":
        return None, "none"

    def _econ(d: dict | None) -> dict:
        if not isinstance(d, dict):
            return {}
        e = d.get("economics")
        return e if isinstance(e, dict) else {}

    # Candidate (margin, price, source) tuples in priority order.
    candidates: list[tuple[float | None, float | None, str]] = [
        (margin_per_kpi, price, "param"),
    ]
    be = _econ(branding)
    candidates.append((be.get("gross_margin"), be.get("price", price), "branding"))
    pe = _econ(preferences)
    candidates.append((pe.get("gross_margin"), pe.get("price", price), "preferences"))

    for margin, pr, source in candidates:
        if margin is None:
            continue
        try:
            m = float(margin)
        except (TypeError, ValueError):
            continue
        if m <= 0:
            continue
        if kind == "units":
            if pr is None:
                continue
            try:
                p = float(pr)
            except (TypeError, ValueError):
                continue
            if p <= 0:
                continue
            return m * p, source
        return m, source
    return None, "none"


# ── Main entry ────────────────────────────────────────────────────────────────


def compute_opportunity_cost(
    mmm: Any,
    design: dict,
    *,
    margin_per_kpi: float | None = None,
    kpi_kind: str = "revenue",
    price: float | None = None,
    preferences: dict | None = None,
    branding: dict | None = None,
    loss_threshold: float | None = None,
    evoi_kpi_units: float | None = None,
    response_horizon_weeks: int | None = None,
    max_draws: int = 200,
    random_seed: int | None = 42,
    contrib_bau: np.ndarray | None = None,
    contrib_exp: np.ndarray | None = None,
    return_draws: bool = False,
) -> OpportunityCostResult:
    """Short-term risk of running ``design`` on the fitted model ``mmm``.

    Args:
        design: a ``planning.design.design_experiment`` payload (carries channel,
            kpi, design_key, intensity_pct / schedule, treatment_geos, duration).
        margin_per_kpi / price / kpi_kind: net-$ valuation inputs. ``revenue``
            multiplies KPI by margin; ``units`` by margin*price; ``other`` reports
            KPI units only. Margin also auto-resolves from ``branding`` /
            ``preferences`` ``economics`` blocks.
        evoi_kpi_units: the experiment's EVOI (planning.evoi, KPI-contribution
            units) for the learning-vs-cost ratio. Pass the EVPI-capped value.
        response_horizon_weeks: denominator for ``evoi_per_week`` (default
            ``mmm.n_periods``).
    """
    channel = design["channel"]
    kpi = design.get("kpi", "")
    design_key = design.get("design_key", "")
    design_type = str(design.get("design_type", design_key))
    duration_requested = int(design.get("duration", 8) or 8)

    (
        treated_mask,
        treated_geo_codes,
        window_codes,
        duration_effective,
        warnings,
    ) = _resolve_treated_rows(mmm, design, duration=duration_requested)

    x_exp, ch_idx, n_test_rows = build_experiment_media(
        mmm, design, treated_mask=treated_mask, window_codes=window_codes
    )
    x_bau = np.asarray(getattr(mmm, "X_media_raw"), dtype=np.float64)

    # Paired counterfactual contributions (F1: Deterministic node, exact
    # pairing). The caller may pass precomputed passes (same mmm/design/max_draws
    # → identical) to avoid recomputing the dominant op cost; ignore a shape
    # mismatch and resample defensively.
    n_obs = x_bau.shape[0]
    if contrib_bau is None or np.asarray(contrib_bau).shape[1] != n_obs:
        contrib_bau = mmm.sample_channel_contributions(
            X_media=x_bau, max_draws=max_draws, random_seed=random_seed
        )  # (D, n_obs, C)
    if contrib_exp is None or np.asarray(contrib_exp).shape[1] != n_obs:
        contrib_exp = mmm.sample_channel_contributions(
            X_media=x_exp, max_draws=max_draws, random_seed=random_seed
        )
    delta_ch = contrib_exp[:, :, ch_idx] - contrib_bau[:, :, ch_idx]  # (D, n_obs)
    n_draws = int(delta_ch.shape[0])

    # Window-only delta (headline; matches the DiD/calibration estimand, F4).
    kpi_delta = delta_ch[:, treated_mask].sum(axis=1)  # (D,)
    # With-carryover: all rows in the treated geos (post-window adstock tail).
    geo_idx = np.asarray(getattr(mmm, "geo_idx"), dtype=np.int64)
    geo_mask_all = np.isin(geo_idx, treated_geo_codes)
    kpi_delta_carry = delta_ch[:, geo_mask_all].sum(axis=1)  # (D,)

    expected_kpi_delta = float(np.mean(kpi_delta))
    kpi_delta_median = float(np.median(kpi_delta))
    kpi_delta_p5 = float(np.percentile(kpi_delta, 5))
    kpi_delta_p95 = float(np.percentile(kpi_delta, 95))
    forgone = np.maximum(0.0, -kpi_delta)
    forgone_kpi_median = float(np.median(forgone))
    forgone_kpi_p95 = float(np.percentile(forgone, 95))
    prob_kpi_loss = float(np.mean(kpi_delta < 0))

    # Observed KPI over the treated window (original scale) for context.
    y_raw = np.asarray(getattr(mmm, "y_raw"), dtype=np.float64)
    window_kpi = float(y_raw[treated_mask].sum())
    pct_of_window_kpi = (
        abs(expected_kpi_delta) / window_kpi if window_kpi > _EPS else None
    )

    # Spend delta — SIGNED, computed internally (never design['weekly_spend_delta']).
    spend_bau_total = float(x_bau[treated_mask, ch_idx].sum())
    spend_exp_total = float(x_exp[treated_mask, ch_idx].sum())
    spend_delta = spend_exp_total - spend_bau_total
    abs_spend_change = abs(spend_delta)
    if design_key == "national_flighting":
        spend_at_risk = 0.0  # budget-neutral
    elif spend_delta < 0:  # holdout: spend you forgo deploying
        spend_at_risk = abs(spend_delta)
    else:  # scaling: extra dollars committed
        spend_at_risk = spend_delta

    # Net-$ business impact (only when a margin resolves).
    margin_per_kpi_val, margin_source = _resolve_margin(
        preferences, branding, margin_per_kpi, kpi_kind, price
    )
    (
        net_profit_median,
        net_profit_p5,
        net_profit_p95,
        oc_dollar_median,
        oc_dollar_p95,
        prob_net_loss,
        prob_loss_over_threshold,
    ) = (None, None, None, None, None, None, None)
    if margin_per_kpi_val is not None:
        net_profit = margin_per_kpi_val * kpi_delta - spend_delta  # (D,) signed
        net_profit_median = float(np.median(net_profit))
        net_profit_p5 = float(np.percentile(net_profit, 5))
        net_profit_p95 = float(np.percentile(net_profit, 95))
        oc_dollar = np.maximum(0.0, -net_profit)
        oc_dollar_median = float(np.median(oc_dollar))
        oc_dollar_p95 = float(np.percentile(oc_dollar, 95))
        prob_net_loss = float(np.mean(net_profit < 0))
        if loss_threshold is not None:
            prob_loss_over_threshold = float(np.mean(oc_dollar > float(loss_threshold)))

    # Learning-vs-cost (single per-week basis).
    horizon = int(response_horizon_weeks or getattr(mmm, "n_periods", 0) or 0)
    cost_per_week = (
        abs(float(np.median(kpi_delta_carry))) / duration_effective
        if duration_effective > 0
        else None
    )
    evoi_per_week = (
        evoi_kpi_units / horizon
        if (evoi_kpi_units is not None and horizon > 0)
        else None
    )
    learning_to_cost_ratio: float | None = None
    if evoi_kpi_units is None:
        learning_to_cost_basis = "unavailable"
    elif evoi_kpi_units <= _EPS:
        learning_to_cost_basis = "channel_already_precise"
    elif cost_per_week is None or cost_per_week <= _EPS:
        learning_to_cost_basis = "net_neutral_design"
    else:
        learning_to_cost_ratio = evoi_per_week / cost_per_week
        learning_to_cost_basis = "kpi_per_week"

    # Honesty flags.
    kpi_scale = max(abs(window_kpi), abs(expected_kpi_delta), 1.0)
    low_information = bool(float(np.std(kpi_delta)) <= kpi_scale * 1e-6)
    ch_hist_max = float(x_bau[:, ch_idx].max())
    extrapolation_warning = bool(
        float(x_exp[treated_mask, ch_idx].max()) > ch_hist_max + _EPS
    )
    notes: list[str] = []
    if low_information:
        notes.append(
            "Near-zero posterior spread on the KPI delta — the intervals are "
            "deceptively tight; treat the point estimate cautiously."
        )
    if extrapolation_warning:
        notes.append(
            "The scaled spend exceeds the channel's observed range — the response "
            "there is extrapolation, not evidence."
        )

    return OpportunityCostResult(
        channel=channel,
        kpi=kpi,
        design_key=design_key,
        design_type=design_type,
        duration_requested=duration_requested,
        duration_effective=duration_effective,
        n_treated_cells=len(treated_geo_codes) * duration_effective,
        n_test_rows=n_test_rows,
        n_draws=n_draws,
        carryover_basis="window_only",
        expected_kpi_delta=expected_kpi_delta,
        kpi_delta_median=kpi_delta_median,
        kpi_delta_p5=kpi_delta_p5,
        kpi_delta_p95=kpi_delta_p95,
        kpi_delta_with_carryover_median=float(np.median(kpi_delta_carry)),
        forgone_kpi_median=forgone_kpi_median,
        forgone_kpi_p95=forgone_kpi_p95,
        prob_kpi_loss=prob_kpi_loss,
        pct_of_window_kpi=pct_of_window_kpi,
        spend_delta=spend_delta,
        abs_spend_change=abs_spend_change,
        spend_at_risk=spend_at_risk,
        margin_per_kpi=margin_per_kpi_val,
        margin_source=margin_source,
        kpi_kind=kpi_kind,
        net_profit_impact_median=net_profit_median,
        net_profit_impact_p5=net_profit_p5,
        net_profit_impact_p95=net_profit_p95,
        opportunity_cost_dollar_median=oc_dollar_median,
        opportunity_cost_dollar_p95=oc_dollar_p95,
        prob_net_loss=prob_net_loss,
        prob_loss_over_threshold=prob_loss_over_threshold,
        loss_threshold=loss_threshold,
        evoi_kpi_units=evoi_kpi_units,
        evoi_per_week=evoi_per_week,
        cost_per_week=cost_per_week,
        learning_to_cost_ratio=learning_to_cost_ratio,
        learning_to_cost_basis=learning_to_cost_basis,
        response_horizon_weeks=horizon or None,
        low_information=low_information,
        extrapolation_warning=extrapolation_warning,
        warnings=warnings,
        notes=notes,
        draws=(
            {"kpi_delta": kpi_delta, "kpi_delta_carry": kpi_delta_carry}
            if return_draws
            else None
        ),
    )
