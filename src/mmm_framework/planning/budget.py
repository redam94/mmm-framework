"""Budget optimization on a fitted BayesianMMM.

Response curves come from the model itself: one posterior-predictive pass per
spend multiplier evaluates the ``channel_contributions`` deterministic with all
channels scaled at once (the model is additive in channels, so each channel's
curve is unaffected by the others' scenario spend). The optimizer then runs in
plain numpy on the sampled curves — greedy marginal allocation, which is exact
for concave (saturating) response curves — and re-optimizing per posterior draw
turns parameter uncertainty into DECISION uncertainty: how stable is the
optimal allocation, not just how wide are the ROIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MULTIPLIERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


@dataclass
class ResponseCurves:
    """Per-channel contribution as a function of spend, with posterior draws.

    contributions[d, c, g] = total (window-summed) original-scale contribution
    of channel c at spend multiplier multipliers[g], under posterior draw d.
    """

    channel_names: list[str]
    multipliers: np.ndarray  # (G,)
    base_spend: np.ndarray  # (C,) current total spend per channel
    contributions: np.ndarray  # (D, C, G)

    @property
    def spend_grid(self) -> np.ndarray:
        """(C, G) actual spend per channel at each multiplier."""
        return self.base_spend[:, None] * self.multipliers[None, :]

    def mean_curves(self) -> np.ndarray:
        """(C, G) posterior-mean contribution curves."""
        return self.contributions.mean(axis=0)


def compute_response_curves(
    mmm: Any,
    multipliers: tuple[float, ...] | None = None,
    max_draws: int = 200,
    random_seed: int | None = None,
) -> ResponseCurves:
    """Sample per-channel spend-response curves from the fitted model.

    One posterior-predictive evaluation per multiplier (NOT per channel ×
    multiplier): all channels are scaled together and read off the additive
    ``channel_contributions`` deterministic.
    """
    mults = np.asarray(multipliers if multipliers is not None else DEFAULT_MULTIPLIERS)
    if 1.0 not in mults:
        mults = np.sort(np.append(mults, 1.0))

    X = mmm.X_media_raw
    base_spend = X.sum(axis=0)

    per_mult = []
    for i, m in enumerate(mults):
        contrib = mmm.sample_channel_contributions(
            X_media=X * float(m),
            max_draws=max_draws,
            random_seed=None if random_seed is None else random_seed + i,
        )  # (D, obs, C)
        per_mult.append(contrib.sum(axis=1))  # (D, C)

    # (G, D, C) -> (D, C, G)
    contributions = np.stack(per_mult, axis=0).transpose(1, 2, 0)
    return ResponseCurves(
        channel_names=list(mmm.channel_names),
        multipliers=mults,
        base_spend=base_spend,
        contributions=contributions,
    )


def _greedy_allocate(
    curves: np.ndarray,  # (C, G) contribution at spend_grid points
    spend_grid: np.ndarray,  # (C, G)
    total_budget: float,
    lo_spend: np.ndarray,  # (C,)
    hi_spend: np.ndarray,  # (C,)
    n_steps: int = 400,
) -> np.ndarray:
    """Allocate ``total_budget`` across channels by repeatedly assigning small
    increments to the channel with the highest marginal contribution per
    dollar (exact for concave curves). Returns spend per channel (C,)."""
    n_channels = curves.shape[0]
    alloc = lo_spend.astype(float).copy()
    remaining = total_budget - alloc.sum()
    if remaining <= 0:
        # Budget can't even cover the lower bounds; scale them proportionally.
        return alloc * (total_budget / max(alloc.sum(), 1e-12))

    step = remaining / n_steps

    def marginal(c: int, s: np.ndarray) -> np.ndarray:
        cur = np.interp(s, spend_grid[c], curves[c])
        nxt = np.interp(s + step, spend_grid[c], curves[c])
        return (nxt - cur) / step

    for _ in range(n_steps):
        gains = np.full(n_channels, -np.inf)
        for c in range(n_channels):
            if alloc[c] + step <= hi_spend[c] + 1e-9:
                gains[c] = marginal(c, alloc[c])
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]):
            break  # every channel at its cap
        alloc[best] += step
    return alloc


def _eval_allocation(
    alloc: np.ndarray, curves: np.ndarray, spend_grid: np.ndarray
) -> float:
    """Total contribution of an allocation under one draw's curves."""
    return float(
        sum(
            np.interp(alloc[c], spend_grid[c], curves[c])
            for c in range(curves.shape[0])
        )
    )


@dataclass
class BudgetOptimizationResult:
    """Optimal allocation plus the decision-uncertainty diagnostics."""

    table: pd.DataFrame  # per-channel: current/optimal spend & share, stability
    total_budget: float
    expected_uplift: float  # median across draws, optimal vs current allocation
    uplift_hdi: tuple[float, float]  # 5–95% interval of the uplift
    prob_positive_uplift: float
    n_draws: int
    notes: list[str] = field(default_factory=list)
    # (D, C) per-draw optimal allocations — the raw decision-uncertainty
    # sample; reused by planning.evoi.compute_evpi to avoid re-optimizing.
    per_draw_alloc: np.ndarray | None = None
    # (C,) the recommended allocation on the mean curves.
    optimal_alloc: np.ndarray | None = None


def optimize_budget(
    mmm: Any = None,
    *,
    curves: ResponseCurves | None = None,
    total_budget: float | None = None,
    budget_change_pct: float | None = None,
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_steps: int = 400,
    max_draws: int = 200,
    random_seed: int | None = None,
) -> BudgetOptimizationResult:
    """Find the budget allocation that maximizes expected KPI contribution.

    Args:
        curves: precomputed response curves (else sampled from ``mmm``).
        total_budget: budget to allocate. Default: current total spend
            (pure reallocation). ``budget_change_pct`` (e.g. ``-10`` or ``15``)
            scales the current total instead.
        min_multiplier / max_multiplier: per-channel spend bounds as multiples
            of CURRENT channel spend (floors/caps keep recommendations inside
            the range the model has evidence for; extrapolating far beyond
            observed spend is curve-fiction).
        bounds: per-channel ``{name: (lo_mult, hi_mult)}`` overrides.
        n_steps: greedy increments (granularity of the allocation).
        max_draws: posterior draws used for curves and stability analysis.
    """
    if curves is None:
        if mmm is None:
            raise ValueError("Provide either a fitted model or precomputed curves.")
        curves = compute_response_curves(
            mmm, max_draws=max_draws, random_seed=random_seed
        )

    names = curves.channel_names
    base = curves.base_spend
    spend_grid = curves.spend_grid
    current_total = float(base.sum())

    if total_budget is None:
        total_budget = current_total
        if budget_change_pct is not None:
            total_budget = current_total * (1.0 + budget_change_pct / 100.0)
    total_budget = float(total_budget)

    lo = np.array(
        [(bounds or {}).get(n, (min_multiplier, max_multiplier))[0] for n in names]
    )
    hi = np.array(
        [(bounds or {}).get(n, (min_multiplier, max_multiplier))[1] for n in names]
    )
    lo_spend, hi_spend = lo * base, hi * base

    notes: list[str] = []
    grid_max = float(curves.multipliers.max())
    if np.any(hi > grid_max):
        notes.append(
            f"Per-channel caps above {grid_max:g}x current spend are clamped to "
            "the sampled curve range."
        )
        hi_spend = np.minimum(hi_spend, base * grid_max)
    if total_budget > hi_spend.sum():
        notes.append(
            "Total budget exceeds the sum of per-channel caps; the surplus "
            "cannot be allocated within bounds."
        )

    mean_curves = curves.mean_curves()
    optimal = _greedy_allocate(
        mean_curves, spend_grid, total_budget, lo_spend, hi_spend, n_steps
    )

    # Decision uncertainty: re-optimize under each draw's curves, and evaluate
    # the recommended vs current allocation under each draw.
    D = curves.contributions.shape[0]
    per_draw_alloc = np.empty((D, len(names)))
    uplift = np.empty(D)
    current_alloc = base.astype(float)
    for d in range(D):
        cd = curves.contributions[d]
        per_draw_alloc[d] = _greedy_allocate(
            cd, spend_grid, total_budget, lo_spend, hi_spend, n_steps
        )
        uplift[d] = _eval_allocation(optimal, cd, spend_grid) - _eval_allocation(
            current_alloc, cd, spend_grid
        )

    share = per_draw_alloc / max(total_budget, 1e-12)
    rows = []
    for c, name in enumerate(names):
        rows.append(
            {
                "channel": name,
                "current_spend": float(base[c]),
                "current_share_pct": 100 * base[c] / max(current_total, 1e-12),
                "optimal_spend": float(optimal[c]),
                "optimal_share_pct": 100 * optimal[c] / max(total_budget, 1e-12),
                "change_pct": 100 * (optimal[c] - base[c]) / max(base[c], 1e-12),
                "optimal_share_p5": float(100 * np.percentile(share[:, c], 5)),
                "optimal_share_p95": float(100 * np.percentile(share[:, c], 95)),
                "allocation_instability": float(
                    np.percentile(share[:, c], 95) - np.percentile(share[:, c], 5)
                ),
            }
        )
    table = pd.DataFrame(rows)

    return BudgetOptimizationResult(
        table=table,
        total_budget=total_budget,
        expected_uplift=float(np.median(uplift)),
        uplift_hdi=(float(np.percentile(uplift, 5)), float(np.percentile(uplift, 95))),
        prob_positive_uplift=float(np.mean(uplift > 0)),
        n_draws=D,
        notes=notes,
        per_draw_alloc=per_draw_alloc,
        optimal_alloc=optimal,
    )


# ── Geo / DMA-level allocation (B4) ───────────────────────────────────────────
# A geo panel exposes a separate response curve per (geography, channel) — each
# observation's contribution depends only on that cell's own (per-cell adstocked)
# media, so scaling every geo's spend by a multiplier and reading one geo's
# summed contribution gives THAT geo's response, independent of the others. We
# build one curve per arm and let the same exact greedy allocator move a single
# national budget across all geo×channel arms at once (a frozen geo or a capped
# channel is just a bound on its arms).

#: Separator between the geo-index prefix and the channel name in a combined arm
#: label. The prefix is the integer geo index (never a geo name) so splitting is
#: unambiguous even if a geography's name contains the separator.
GEO_ARM_SEP = " │ "


def compute_response_curves_per_geo(
    mmm: Any,
    multipliers: tuple[float, ...] | None = None,
    max_draws: int = 200,
    random_seed: int | None = None,
) -> dict[str, ResponseCurves]:
    """Per-geography spend-response curves from a fitted geo panel.

    One posterior-predictive evaluation per multiplier (all geos scaled together,
    as in :func:`compute_response_curves`); contributions are then grouped by
    ``mmm.geo_idx`` so each geography gets its own :class:`ResponseCurves` over
    that geography's current spend. Requires ``mmm.has_geo``.
    """
    if not getattr(mmm, "has_geo", False) or int(getattr(mmm, "n_geos", 1)) <= 1:
        raise ValueError(
            "Model has no geo dimension — use compute_response_curves for national."
        )
    mults = np.asarray(multipliers if multipliers is not None else DEFAULT_MULTIPLIERS)
    if 1.0 not in mults:
        mults = np.sort(np.append(mults, 1.0))

    X = mmm.X_media_raw
    geo_idx = np.asarray(mmm.geo_idx)
    geo_names = list(mmm.geo_names)
    n_geos = len(geo_names)

    per_mult_by_geo: dict[int, list[np.ndarray]] = {g: [] for g in range(n_geos)}
    for i, m in enumerate(mults):
        contrib = mmm.sample_channel_contributions(
            X_media=X * float(m),
            max_draws=max_draws,
            random_seed=None if random_seed is None else random_seed + i,
        )  # (D, n_obs, C)
        for g in range(n_geos):
            mask = geo_idx == g
            per_mult_by_geo[g].append(contrib[:, mask, :].sum(axis=1))  # (D, C)

    out: dict[str, ResponseCurves] = {}
    for g, name in enumerate(geo_names):
        mask = geo_idx == g
        base_spend = X[mask].sum(axis=0)  # (C,)
        # (G, D, C) -> (D, C, G)
        contributions = np.stack(per_mult_by_geo[g], axis=0).transpose(1, 2, 0)
        out[name] = ResponseCurves(
            channel_names=list(mmm.channel_names),
            multipliers=mults,
            base_spend=base_spend,
            contributions=contributions,
        )
    return out


def combine_geo_curves(geo_curves: dict[str, ResponseCurves]) -> ResponseCurves:
    """Flatten ``{geo: ResponseCurves}`` into a single :class:`ResponseCurves`
    whose "channels" are ``"{geo_index} │ {channel}"`` arms, so the national
    greedy optimizer can allocate one budget jointly across every geo×channel
    arm. All inputs must share the same multipliers and channel ordering."""
    geos = list(geo_curves)
    if not geos:
        raise ValueError("No geo curves to combine.")
    mults = geo_curves[geos[0]].multipliers
    arm_names: list[str] = []
    base: list[float] = []
    contribs: list[np.ndarray] = []
    for g_idx, g in enumerate(geos):
        rc = geo_curves[g]
        for c, ch in enumerate(rc.channel_names):
            arm_names.append(f"{g_idx}{GEO_ARM_SEP}{ch}")
            base.append(float(rc.base_spend[c]))
            contribs.append(rc.contributions[:, c, :])  # (D, G_mult)
    contributions = np.stack(contribs, axis=1)  # (D, A, G_mult)
    return ResponseCurves(
        channel_names=arm_names,
        multipliers=mults,
        base_spend=np.asarray(base),
        contributions=contributions,
    )


def optimize_budget_by_geo(
    mmm: Any,
    *,
    total_budget: float | None = None,
    budget_change_pct: float | None = None,
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_steps: int = 400,
    max_draws: int = 200,
    random_seed: int | None = None,
) -> BudgetOptimizationResult:
    """Allocate one national budget jointly across every (geo, channel) arm.

    Builds per-geo curves, flattens them to arms, and reuses :func:`optimize_budget`
    so the greedy marginal allocator, per-draw stability, and uplift diagnostics
    are identical to the national path. The returned ``table`` gains a ``geo``
    column and the ``channel`` column holds the bare channel name. Per-channel
    ``bounds`` apply to that channel in **every** geography.
    """
    geo_curves = compute_response_curves_per_geo(
        mmm, max_draws=max_draws, random_seed=random_seed
    )
    combined = combine_geo_curves(geo_curves)

    arm_bounds: dict[str, tuple[float, float]] | None = None
    if bounds:
        arm_bounds = {}
        for arm in combined.channel_names:
            ch = arm.split(GEO_ARM_SEP, 1)[1]
            if ch in bounds:
                arm_bounds[arm] = bounds[ch]

    res = optimize_budget(
        curves=combined,
        total_budget=total_budget,
        budget_change_pct=budget_change_pct,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        bounds=arm_bounds,
        n_steps=n_steps,
        max_draws=max_draws,
        random_seed=random_seed,
    )

    geo_names = list(mmm.geo_names)
    t = res.table.copy()
    split = t["channel"].str.split(GEO_ARM_SEP, n=1, expand=True)
    t.insert(0, "geo", [geo_names[int(g)] for g in split[0]])
    t["channel"] = split[1].to_numpy()
    res.table = t
    return res
