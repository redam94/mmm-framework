"""The EIG/EVOI experiment-priority grid.

Combines EIG (what an experiment would teach — planning.eig) with EVOI (what
that learning is worth to the budget decision — planning.evoi) into a
per-channel priority grid with the 2×2 quadrant classification:

    high EIG, high EVOI → test_now        (decision-critical uncertainty)
    high EIG, low EVOI  → learn_cheaply   (informative, but the decision is robust)
    low EIG, high EVOI  → monitor         (high stakes, already precise — watch drift)
    low EIG, low EVOI   → deprioritize

Composite score: priority = sqrt(eig_norm * evoi_norm) — a normalized
geometric mean, so a channel cannot win on one axis alone.

When the experiment registry supplies per-channel evidence dates, information
decay (planning.eig) yields the decayed EIG and the re-test trigger.

Import-light (numpy/pandas only) so it can run inside the session kernels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from .budget import (
    BudgetOptimizationResult,
    ResponseCurves,
    compute_response_curves,
    optimize_budget,
)
from .eig import (
    DEFAULT_RETEST_THRESHOLD_NATS,
    channel_half_life,
    eig_gaussian,
    eig_monte_carlo,
    reexperiment_due,
    sigma_exp_for_design,
    use_gaussian,
)
from .evoi import compute_evoi_for_channel, compute_evpi

QUADRANTS = {
    ("high", "high"): "test_now",
    ("high", "low"): "learn_cheaply",
    ("low", "high"): "monitor",
    ("low", "low"): "deprioritize",
}


@dataclass
class ChannelPriority:
    """One row of the priority grid (JSON-safe via .to_dict())."""

    channel: str
    spend: float
    spend_share: float
    roi_mean: float
    roi_sd: float
    roi_hdi_low: float
    roi_hdi_high: float
    sigma_exp: float
    design_type: str
    eig: float
    eig_method: str  # "gaussian" | "monte_carlo"
    evoi: float
    evoi_pct_budget: float
    evpi_share: float  # evoi / evpi
    priority: float  # composite sqrt(eig_norm * evoi_norm)
    quadrant: str
    weeks_since_evidence: float | None = None
    eig_decayed: float | None = None
    retest_due: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s)).date()
    except Exception:
        return None


def compute_experiment_priorities(
    mmm: Any = None,
    *,
    curves: ResponseCurves | None = None,
    optimization: BudgetOptimizationResult | None = None,
    evidence: dict[str, dict] | None = None,
    as_of: str | None = None,
    design_type: str | None = None,
    sigma_exp_overrides: dict[str, float] | None = None,
    half_life_overrides: dict[str, float] | None = None,
    eig_threshold: float | None = None,
    evoi_threshold: float | None = None,
    retest_threshold_nats: float = DEFAULT_RETEST_THRESHOLD_NATS,
    n_outcomes: int = 48,
    # Must match optimize_budget's allocation granularity: v_current is valued
    # at the optimizer's allocation, and a coarser per-outcome reallocation
    # would bias every EVOI difference negative.
    n_steps: int = 400,
    max_draws: int = 200,
    random_seed: int | None = 42,
) -> tuple[list[ChannelPriority], dict[str, Any]]:
    """Per-channel EIG/EVOI priority grid + portfolio summary.

    Args:
        evidence: ``{channel: {"end_date": ISO, ...}}`` — the newest calibrated
            experiment per channel (``sessions.latest_calibrated_evidence``).
            Drives information decay and the re-test trigger.
        as_of: ISO date for the decay clock (default: today).
        design_type: assumed experiment design for sigma_exp; default
            ``geo_holdout`` when the model has geo structure, else
            ``national_pulse``.
        eig_threshold / evoi_threshold: absolute quadrant splits; default is
            the median of each axis across channels.

    Returns ``(grid, portfolio)`` where portfolio carries v_current, EVPI,
    total budget, draws, and the thresholds used.
    """
    if curves is None:
        if mmm is None:
            raise ValueError("Provide either a fitted model or precomputed curves.")
        curves = compute_response_curves(
            mmm, max_draws=max_draws, random_seed=random_seed
        )
    if optimization is None:
        optimization = optimize_budget(curves=curves, random_seed=random_seed)

    names = curves.channel_names
    base = curves.base_spend.astype(float)
    total_spend = float(base.sum())
    g1 = int(np.argmin(np.abs(curves.multipliers - 1.0)))
    contrib_at_current = curves.contributions[:, :, g1]  # (D, C)
    D = contrib_at_current.shape[0]

    if design_type is None:
        design_type = (
            "geo_holdout" if bool(getattr(mmm, "has_geo", False)) else "national_pulse"
        )

    port = compute_evpi(
        curves,
        total_budget=optimization.total_budget,
        per_draw_alloc=optimization.per_draw_alloc,
        optimal_alloc=optimization.optimal_alloc,
        n_steps=n_steps,
    )

    # Common random numbers: the same outcome indices + noise for every channel
    # so MC noise can't rank-flip near-ties.
    rng = np.random.default_rng(random_seed)
    d_idx = rng.integers(0, D, size=n_outcomes)
    z = rng.standard_normal(n_outcomes)

    as_of_date = _parse_date(as_of) or date.today()

    raw: list[dict[str, Any]] = []
    for c, name in enumerate(names):
        spend = float(base[c])
        roi_draws = contrib_at_current[:, c] / max(spend, 1e-12)
        roi_mean = float(np.mean(roi_draws))
        roi_sd = float(np.std(roi_draws))
        p5, p95 = (float(np.percentile(roi_draws, q)) for q in (5, 95))
        sigma_exp = (sigma_exp_overrides or {}).get(name) or sigma_exp_for_design(
            design_type, float(np.median(roi_draws))
        )

        if use_gaussian(roi_draws):
            eig, eig_method = eig_gaussian(roi_sd, sigma_exp), "gaussian"
        else:
            eig, eig_method = (
                eig_monte_carlo(
                    roi_draws, sigma_exp, n_outcomes=max(64, n_outcomes), rng=rng
                ),
                "monte_carlo",
            )

        evoi = compute_evoi_for_channel(
            curves,
            c,
            roi_draws,
            sigma_exp,
            optimal_alloc=optimization.optimal_alloc,
            total_budget=optimization.total_budget,
            n_steps=n_steps,
            outcome_draws=(d_idx, z),
        )
        evoi = min(evoi, port.evpi) if port.evpi > 0 else evoi

        # Information decay against the newest calibrated evidence
        weeks_since: float | None = None
        eig_dec: float | None = None
        retest = False
        ev = (evidence or {}).get(name)
        ev_date = _parse_date(ev.get("end_date")) if ev else None
        if ev_date is not None:
            weeks_since = max((as_of_date - ev_date).days, 0) / 7.0
            hl = channel_half_life(name, half_life_overrides)
            retest, eig_dec = reexperiment_due(
                roi_sd,
                weeks_since,
                hl,
                sigma_exp,
                threshold_nats=retest_threshold_nats,
            )

        raw.append(
            dict(
                channel=name,
                spend=spend,
                spend_share=spend / max(total_spend, 1e-12),
                roi_mean=roi_mean,
                roi_sd=roi_sd,
                roi_hdi_low=p5,
                roi_hdi_high=p95,
                sigma_exp=float(sigma_exp),
                design_type=design_type,
                eig=float(eig),
                eig_method=eig_method,
                evoi=float(evoi),
                evoi_pct_budget=100 * evoi / max(optimization.total_budget, 1e-12),
                evpi_share=evoi / port.evpi if port.evpi > 0 else 0.0,
                weeks_since_evidence=weeks_since,
                eig_decayed=eig_dec,
                retest_due=retest,
            )
        )

    eigs = np.array([r["eig"] for r in raw])
    evois = np.array([r["evoi"] for r in raw])
    max_eig = float(eigs.max()) if eigs.size and eigs.max() > 0 else 1.0
    max_evoi = float(evois.max()) if evois.size and evois.max() > 0 else 1.0
    eig_split = float(np.median(eigs)) if eig_threshold is None else eig_threshold
    evoi_split = float(np.median(evois)) if evoi_threshold is None else evoi_threshold

    grid: list[ChannelPriority] = []
    for r in raw:
        quad = QUADRANTS[
            (
                "high" if r["eig"] >= eig_split else "low",
                "high" if r["evoi"] >= evoi_split else "low",
            )
        ]
        grid.append(
            ChannelPriority(
                priority=float(np.sqrt((r["eig"] / max_eig) * (r["evoi"] / max_evoi))),
                quadrant=quad,
                **r,
            )
        )
    grid.sort(key=lambda g: g.priority, reverse=True)

    portfolio = {
        "v_current": port.v_current,
        "evpi": port.evpi,
        "total_budget": optimization.total_budget,
        "total_spend": total_spend,
        "n_draws": D,
        "n_outcomes": n_outcomes,
        "design_type": design_type,
        "eig_threshold": eig_split,
        "evoi_threshold": evoi_split,
        "retest_threshold_nats": retest_threshold_nats,
        "as_of": as_of_date.isoformat(),
    }
    return grid, portfolio
