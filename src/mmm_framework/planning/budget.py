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

#: A recommendation may exceed ``max_obs_multiplier`` by this fraction before it
#: is flagged as extrapolating beyond observed spend (issue #105).
_EXTRAP_TOL = 0.02
#: Epistemic honesty margin: the recommended-spend credible interval is widened
#: by ``1 + k·(mult − max_obs_multiplier)`` beyond observed support, because the
#: posterior parameter spread there understates true (model-form) uncertainty —
#: the saturation FORM itself may be wrong past the data.
_EXTRAP_INFLATION_K = 0.6

#: Default per-channel deviation cap for a *default reallocation* (the plan shown
#: in a client report when the user has not run the Planner studio). Each
#: channel may move at most ±20% from its current spend so no channel is turned
#: off completely and every recommendation stays inside the spend range the model
#: has direct evidence for. See :func:`default_reallocation`.
DEFAULT_REALLOC_DEVIATION = 0.20


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
    # Per-channel observed per-observation spend range (issue #105). The response
    # curve is data-supported only up to the largest single-period spend the
    # model actually saw; a recommendation that scales current spend past
    # ``max_obs_multiplier`` pushes the AVERAGE period beyond any observed level
    # → extrapolation. ``None`` for curves built without a model (back-compat).
    obs_max_spend: np.ndarray | None = None  # (C,) max spend at any observation
    # Observations behind ``base_spend``: a single int for national curves, or a
    # per-channel ``(C,)`` array for combined geo arms whose geographies may have
    # different period counts (ragged panels) — see ``combine_geo_curves``.
    n_obs: int | np.ndarray | None = None

    @property
    def spend_grid(self) -> np.ndarray:
        """(C, G) actual spend per channel at each multiplier."""
        return self.base_spend[:, None] * self.multipliers[None, :]

    @property
    def max_obs_multiplier(self) -> np.ndarray | None:
        """(C,) the largest CURRENT-spend multiple that stays within observed
        support: ``max_per_obs_spend / mean_per_obs_spend``. Scaling current
        spend past this pushes the average period beyond anything observed, so
        the response there is the saturation FORM extrapolating. ``None`` when
        the observed range was not captured."""
        if self.obs_max_spend is None or self.n_obs is None:
            return None
        # np.maximum handles both a scalar n_obs (national) and a per-arm array
        # (combined geo arms) without a Python truth-value ambiguity.
        n = np.maximum(np.asarray(self.n_obs, dtype=float), 1.0)
        mean_per_obs = self.base_spend / n
        with np.errstate(divide="ignore", invalid="ignore"):
            m = self.obs_max_spend / np.where(mean_per_obs > 0, mean_per_obs, np.nan)
        # A channel run at a perfectly constant level has max==mean → 1.0 (any
        # scale-up is extrapolation); a spiky channel earns more headroom.
        return np.where(np.isfinite(m), np.maximum(m, 1.0), 1.0)

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

    X = np.asarray(mmm.X_media_raw, dtype=float)
    base_spend = X.sum(axis=0)
    obs_max_spend = X.max(axis=0) if X.ndim == 2 and X.shape[0] else None
    n_obs = int(X.shape[0]) if X.ndim == 2 else None

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
        obs_max_spend=obs_max_spend,
        n_obs=n_obs,
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


# ---------------------------------------------------------------------------
# Budget optimizer v2 (#139): risk objectives, constrained SLSQP solve, and
# breakeven / free-budget mode with shadow prices.
# ---------------------------------------------------------------------------

#: Recognized risk objectives for the response curve the optimizer maximizes.
_OBJECTIVES_DOC = (
    "'mean' (posterior mean), 'p<q>' (downside q-th percentile, e.g. 'p10'), or "
    "'cvar<a>' (mean of the worst a% of draws, e.g. 'cvar5')"
)


def objective_curves(curves: ResponseCurves, objective: str = "mean") -> np.ndarray:
    """(C, G) response curve for a risk objective — the curve the allocator
    maximizes. ``'mean'`` = posterior mean (risk-neutral); ``'p<q>'`` = the
    per-channel q-th percentile across draws (a conservative *downside* curve);
    ``'cvar<a>'`` = the per-channel mean of the worst a% of draws (expected
    shortfall). Per-channel quantiles are a conservative proxy for the joint
    portfolio quantile (they ignore cross-channel draw correlation), documented
    in ``technical-docs/budget-optimizer-v2.md``.
    """
    c = curves.contributions  # (D, C, G)
    obj = (objective or "mean").strip().lower()
    if obj in ("mean", "expected"):
        return c.mean(axis=0)
    if obj.startswith("cvar"):
        a = float(obj[4:] or 5.0)
        if not 0 < a < 100:
            raise ValueError(f"cvar level must be in (0, 100): got {a}")
        D = c.shape[0]
        k = max(1, int(np.ceil(a / 100.0 * D)))
        return np.sort(c, axis=0)[:k].mean(axis=0)  # worst-k mean
    if obj.startswith("p") and obj[1:].replace(".", "", 1).isdigit():
        q = float(obj[1:])
        if not 0 <= q <= 100:
            raise ValueError(f"percentile must be in [0, 100]: got {q}")
        return np.percentile(c, q, axis=0)
    raise ValueError(f"Unknown objective '{objective}'. Use one of: {_OBJECTIVES_DOC}.")


def objective_label(objective: str = "mean") -> str:
    """Human-readable label for an objective, for reporting which one was used."""
    obj = (objective or "mean").strip().lower()
    if obj in ("mean", "expected"):
        return "expected KPI (posterior mean)"
    if obj.startswith("cvar"):
        return f"CVaR{obj[4:] or '5'} (mean of the worst draws — risk-averse)"
    if obj.startswith("p"):
        return f"downside P{obj[1:]} (a low quantile — risk-averse)"
    return objective


def _segment_marginal(curve_c: np.ndarray, grid_c: np.ndarray, s: float) -> float:
    """Local marginal contribution per dollar at spend ``s`` on one channel's
    piecewise-linear curve (the slope of the segment containing ``s``; the last
    segment's slope past the grid)."""
    j = int(np.searchsorted(grid_c, s, side="right")) - 1
    j = max(0, min(j, len(grid_c) - 2))
    dx = grid_c[j + 1] - grid_c[j]
    if dx <= 0:
        return 0.0
    return float((curve_c[j + 1] - curve_c[j]) / dx)


def _normalize_groups(
    groups: list[dict] | None, names: list[str], total_budget: float
) -> list[dict]:
    """Convert user group specs to internal ``{indices, min_spend, max_spend}``
    (absolute dollars). A group names ``channels`` and either share bounds
    (``min_share`` / ``max_share``, fractions of the total budget) or absolute
    dollar bounds (``min_spend`` / ``max_spend``). Unknown channel names raise."""
    idx = {n: c for c, n in enumerate(names)}
    out: list[dict] = []
    for g in groups or []:
        chans = g.get("channels") or []
        indices = []
        for ch in chans:
            if ch not in idx:
                raise ValueError(
                    f"group '{g.get('name', '?')}': unknown channel '{ch}'"
                )
            indices.append(idx[ch])
        entry: dict = {"indices": indices, "name": g.get("name")}
        if g.get("min_share") is not None:
            entry["min_spend"] = float(g["min_share"]) * total_budget
        if g.get("max_share") is not None:
            entry["max_spend"] = float(g["max_share"]) * total_budget
        if g.get("min_spend") is not None:
            entry["min_spend"] = float(g["min_spend"])
        if g.get("max_spend") is not None:
            entry["max_spend"] = float(g["max_spend"])
        out.append(entry)
    return out


def _solve_allocation(
    curve: np.ndarray,  # (C, G) objective curve
    spend_grid: np.ndarray,  # (C, G)
    *,
    total_budget: float | None,
    lo_spend: np.ndarray,  # (C,)
    hi_spend: np.ndarray,  # (C,)
    groups: list[dict] | None = None,
    mode: str = "fixed",
    value_per_kpi: float = 1.0,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Constrained allocation on one response curve via SLSQP (convex: a sum of
    concave channel curves under linear constraints).

    ``mode='fixed'`` spends exactly ``total_budget``; ``mode='free'`` (breakeven)
    treats the total as a decision and funds each channel until its marginal
    return hits the breakeven line — maximizing ``value_per_kpi·KPI − spend``.

    ``groups`` is a list of ``{"indices": [...], "min_spend"/"max_spend": $}``
    portfolio constraints (absolute dollars; share bounds are converted upstream).
    Returns ``(alloc (C,), shadow_price, marginal_roas (C,))`` where the shadow
    price is the budget constraint's water level (the marginal KPI-value of the
    next dollar) and ``marginal_roas[c] = value_per_kpi · dKPI/dspend`` at the
    solution (the funding line: fund while it exceeds 1).
    """
    from scipy.optimize import minimize

    C = curve.shape[0]

    def value(s: np.ndarray) -> float:
        return float(sum(np.interp(s[c], spend_grid[c], curve[c]) for c in range(C)))

    def marginals(s: np.ndarray) -> np.ndarray:
        return np.array(
            [_segment_marginal(curve[c], spend_grid[c], s[c]) for c in range(C)]
        )

    if mode == "free":
        # Maximize profit value_per_kpi·KPI(s) − Σs → minimize the negative.
        def neg(s):
            return -(value_per_kpi * value(s) - float(s.sum()))

        def neg_jac(s):
            return -(value_per_kpi * marginals(s) - 1.0)

        constraints = []
    else:  # fixed budget

        def neg(s):
            return -value_per_kpi * value(s)

        def neg_jac(s):
            return -value_per_kpi * marginals(s)

        constraints = [
            {
                "type": "eq",
                "fun": lambda s: float(s.sum()) - float(total_budget),
                "jac": lambda s: np.ones(C),
            }
        ]

    for g in groups or []:
        idx = list(g["indices"])
        sel = np.zeros(C)
        sel[idx] = 1.0
        if g.get("min_spend") is not None:
            lo_g = float(g["min_spend"])
            constraints.append(
                {
                    "type": "ineq",  # Σ s[idx] − lo ≥ 0
                    "fun": (lambda s, sel=sel, lo=lo_g: float(sel @ s) - lo),
                    "jac": (lambda s, sel=sel: sel),
                }
            )
        if g.get("max_spend") is not None:
            hi_g = float(g["max_spend"])
            constraints.append(
                {
                    "type": "ineq",  # hi − Σ s[idx] ≥ 0
                    "fun": (lambda s, sel=sel, hi=hi_g: hi - float(sel @ s)),
                    "jac": (lambda s, sel=sel: -sel),
                }
            )

    bnds = [
        (float(lo_spend[c]), float(max(hi_spend[c], lo_spend[c]))) for c in range(C)
    ]
    if x0 is None:
        if mode == "free":
            x0 = np.clip(0.5 * (lo_spend + hi_spend), lo_spend, hi_spend)
        else:
            # feasible warm start: lower bounds + budget spread proportionally.
            x0 = lo_spend.astype(float).copy()
            extra = max(float(total_budget) - x0.sum(), 0.0)
            room = np.maximum(hi_spend - lo_spend, 0.0)
            if room.sum() > 0:
                x0 = x0 + extra * room / room.sum()
    res = minimize(
        neg,
        np.asarray(x0, dtype=float),
        jac=neg_jac,
        method="SLSQP",
        bounds=bnds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    alloc = np.clip(res.x, lo_spend, np.maximum(hi_spend, lo_spend))
    if mode != "free" and total_budget and alloc.sum() > 0:
        # SLSQP may leave a small equality residual; renormalize to the budget
        # within bounds (proportional to headroom above the floor).
        room = np.maximum(alloc - lo_spend, 1e-12)
        deficit = float(total_budget) - alloc.sum()
        alloc = alloc + deficit * room / room.sum()
        alloc = np.clip(alloc, lo_spend, np.maximum(hi_spend, lo_spend))
    m = value_per_kpi * marginals(alloc)
    funded = (alloc > lo_spend + 1e-9) & (alloc < hi_spend - 1e-9)
    shadow = float(np.median(m[funded])) if funded.any() else float(np.max(m))
    return alloc, shadow, m


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
    # Expected KPI left on the table by committing to this single plan under
    # parameter uncertainty (mean over draws of each draw's own-optimal minus the
    # recommended plan, floored at 0) — the "expected regret" headline (issue #105).
    expected_regret: float = 0.0
    # Number of channels whose recommendation extrapolates past observed spend.
    n_extrapolated: int = 0
    # Budget optimizer v2 (#139). The objective the allocator maximized ("mean",
    # "p10", "cvar5", …) and its human label; the optimization mode ("fixed" =
    # spend the budget, "free" = fund to breakeven); the budget shadow price (the
    # marginal KPI-value of the next dollar — the water level greedy equalizes);
    # and per-channel marginal ROAS at the recommended plan (the funding line).
    objective: str = "mean"
    objective_label: str = "expected KPI (posterior mean)"
    mode: str = "fixed"
    shadow_price: float | None = None
    marginal_roas: dict[str, float] | None = None


def optimize_budget(
    mmm: Any = None,
    *,
    curves: ResponseCurves | None = None,
    total_budget: float | None = None,
    budget_change_pct: float | None = None,
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    bounds: dict[str, tuple[float, float]] | None = None,
    abs_bounds: dict[str, tuple[float, float]] | None = None,
    groups: list[dict] | None = None,
    min_channel_spend: float | dict[str, float] | None = None,
    objective: str = "mean",
    mode: str = "fixed",
    value_per_kpi: float = 1.0,
    n_steps: int = 400,
    max_draws: int = 200,
    random_seed: int | None = None,
) -> BudgetOptimizationResult:
    """Find the budget allocation that maximizes the chosen KPI objective.

    Args:
        curves: precomputed response curves (else sampled from ``mmm``).
        total_budget: budget to allocate. Default: current total spend
            (pure reallocation). ``budget_change_pct`` (e.g. ``-10`` or ``15``)
            scales the current total instead. Ignored in ``mode="free"``.
        min_multiplier / max_multiplier: per-channel spend bounds as multiples
            of CURRENT channel spend (floors/caps keep recommendations inside
            the range the model has evidence for; extrapolating far beyond
            observed spend is curve-fiction).
        bounds: per-channel ``{name: (lo_mult, hi_mult)}`` multiplier overrides.
        abs_bounds: per-channel ``{name: (lo_$, hi_$)}`` ABSOLUTE dollar bounds
            (override the multiplier bounds for that channel) — #139.
        groups: portfolio constraints, a list of
            ``{"name", "channels": [...], "min_share"/"max_share": frac,
            "min_spend"/"max_spend": $}`` — e.g. "digital ≥ 40%" — #139.
        min_channel_spend: keep-every-channel-on floor (a scalar applied to all
            channels, or per-channel ``{name: $}``) — #139.
        objective: the risk objective the allocator maximizes — 'mean'
            (posterior mean), 'p<q>' (downside q-th percentile, e.g. 'p10'), or
            'cvar<a>' (mean of the worst a% of draws, e.g. 'cvar5') (#139).
        mode: ``"fixed"`` spends the budget; ``"free"`` funds each channel to
            breakeven (marginal value = $1), the total becomes an output (#139).
        value_per_kpi: $ value of one KPI unit (for breakeven / marginal ROAS
            when the KPI is not already revenue). Default 1.0.
        n_steps: greedy increments (granularity of the default allocation).
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

    # #139 absolute-$ bounds override the multiplier-derived bounds per channel.
    if abs_bounds:
        idx = {n: c for c, n in enumerate(names)}
        for n, (blo, bhi) in abs_bounds.items():
            if n in idx:
                lo_spend[idx[n]] = float(blo)
                hi_spend[idx[n]] = min(float(bhi), base[idx[n]] * grid_max)

    # #139 keep-on floor: raise each channel's lower bound so no channel is off.
    if min_channel_spend is not None:
        for c, n in enumerate(names):
            floor = (
                float(min_channel_spend.get(n, 0.0))
                if isinstance(min_channel_spend, dict)
                else float(min_channel_spend)
            )
            lo_spend[c] = max(lo_spend[c], min(floor, hi_spend[c]))

    # #139 normalize group share constraints → absolute-$ against the budget.
    norm_groups = _normalize_groups(groups, names, total_budget) if groups else None

    if mode != "free" and total_budget > hi_spend.sum():
        notes.append(
            "Total budget exceeds the sum of per-channel caps; the surplus "
            "cannot be allocated within bounds."
        )

    # Choose the allocator. The historical greedy path (mean objective, fixed
    # mode, no advanced constraints) is preserved byte-for-byte; any v2 feature
    # routes through the constrained SLSQP solver.
    advanced = (
        objective != "mean"
        or mode != "fixed"
        or bool(norm_groups)
        or bool(abs_bounds)
        or min_channel_spend is not None
    )
    obj_curve = objective_curves(curves, objective)  # (C, G)
    obj_label = objective_label(objective)

    def _allocate(curve2d: np.ndarray, budget: float, x0=None) -> np.ndarray:
        if not advanced:
            return _greedy_allocate(
                curve2d, spend_grid, budget, lo_spend, hi_spend, n_steps
            )
        alloc, _, _ = _solve_allocation(
            curve2d,
            spend_grid,
            total_budget=budget,
            lo_spend=lo_spend,
            hi_spend=hi_spend,
            groups=norm_groups,
            mode=mode,
            value_per_kpi=value_per_kpi,
            x0=x0,
        )
        return alloc

    mean_curves = obj_curve
    optimal = _allocate(mean_curves, total_budget)
    if mode == "free":
        total_budget = float(optimal.sum())  # breakeven total is an output
    # Shadow price + marginal ROAS AT the recommended plan (v2 readouts): the
    # per-channel marginal value of the next dollar, and the budget water level
    # (the level greedy/SLSQP equalizes across interior funded channels).
    marg = value_per_kpi * np.array(
        [
            _segment_marginal(mean_curves[c], spend_grid[c], optimal[c])
            for c in range(len(names))
        ]
    )
    interior = (optimal > lo_spend + 1e-9) & (optimal < hi_spend - 1e-9)
    shadow_price = (
        float(np.median(marg[interior])) if interior.any() else float(np.max(marg))
    )
    marginal_roas = {n: float(marg[c]) for c, n in enumerate(names)}

    # Decision uncertainty: re-optimize under each draw's curves, and evaluate
    # the recommended vs current allocation under each draw.
    D = curves.contributions.shape[0]
    per_draw_alloc = np.empty((D, len(names)))
    uplift = np.empty(D)
    # For expected regret: the value of the recommended (point) plan vs the value
    # of THIS draw's own optimal, both under the draw's curves. Their gap is the
    # KPI left on the table by committing to one plan under parameter uncertainty.
    v_plan = np.empty(D)
    v_perfect = np.empty(D)
    current_alloc = base.astype(float)
    for d in range(D):
        cd = curves.contributions[d]
        per_draw_alloc[d] = _allocate(cd, total_budget, x0=optimal)
        v_plan[d] = _eval_allocation(optimal, cd, spend_grid)
        v_perfect[d] = _eval_allocation(per_draw_alloc[d], cd, spend_grid)
        uplift[d] = v_plan[d] - _eval_allocation(current_alloc, cd, spend_grid)

    # Observed-support boundary per channel (issue #105): scale-up beyond this
    # multiple of current spend pushes the average period past anything observed.
    max_obs_mult = curves.max_obs_multiplier  # (C,) or None
    n_extrapolated = 0

    rows = []
    for c, name in enumerate(names):
        alloc_c = per_draw_alloc[:, c]
        share_c = alloc_c / max(total_budget, 1e-12)
        spend_p5 = float(np.percentile(alloc_c, 5))
        spend_p95 = float(np.percentile(alloc_c, 95))
        opt_mult = float(optimal[c] / max(base[c], 1e-12))
        m_obs = (
            float(max_obs_mult[c])
            if max_obs_mult is not None and np.isfinite(max_obs_mult[c])
            else None
        )
        within = True if m_obs is None else opt_mult <= m_obs * (1.0 + _EXTRAP_TOL)
        if not within:
            n_extrapolated += 1
            # Widen the recommended-spend CI to reflect the extra (model-form)
            # uncertainty of extrapolating past observed spend.
            infl = 1.0 + _EXTRAP_INFLATION_K * max(0.0, opt_mult - (m_obs or opt_mult))
            med = float(optimal[c])
            spend_p5 = med - (med - spend_p5) * infl
            spend_p95 = med + (spend_p95 - med) * infl
        rows.append(
            {
                "channel": name,
                "current_spend": float(base[c]),
                "current_share_pct": 100 * base[c] / max(current_total, 1e-12),
                "optimal_spend": float(optimal[c]),
                "optimal_share_pct": 100 * optimal[c] / max(total_budget, 1e-12),
                "change_pct": 100 * (optimal[c] - base[c]) / max(base[c], 1e-12),
                "optimal_spend_p5": spend_p5,
                "optimal_spend_p95": spend_p95,
                "optimal_share_p5": float(100 * np.percentile(share_c, 5)),
                "optimal_share_p95": float(100 * np.percentile(share_c, 95)),
                "allocation_instability": float(
                    np.percentile(share_c, 95) - np.percentile(share_c, 5)
                ),
                "within_observed_range": bool(within),
                "max_obs_multiplier": m_obs,
                "recommended_multiplier": opt_mult,
            }
        )
    table = pd.DataFrame(rows)

    if n_extrapolated:
        notes.append(
            f"{n_extrapolated} channel(s) are recommended beyond the spend range "
            "the model has observed; those figures extrapolate the response curve "
            "(their intervals are widened accordingly). Confirm with a test before "
            "committing large scale-ups."
        )

    expected_regret = float(np.mean(np.maximum(v_perfect - v_plan, 0.0)))

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
        expected_regret=expected_regret,
        n_extrapolated=n_extrapolated,
        objective=objective,
        objective_label=obj_label,
        mode=mode,
        shadow_price=shadow_price,
        marginal_roas=marginal_roas,
    )


def _result_to_report_dict(
    res: BudgetOptimizationResult, *, deviation: float | None = None
) -> dict[str, Any]:
    """Flatten a :class:`BudgetOptimizationResult` into the report-ready dict the
    ``AllocationSection`` / ``AugurAllocationSection`` / ``plan_budget`` op all
    consume (``allocation`` rows + headline uplift fields)."""
    t = res.table
    has_range = "within_observed_range" in t.columns

    def _row(r: "pd.Series") -> dict[str, Any]:
        out = {
            "channel": str(r["channel"]),
            "current_spend": float(r["current_spend"]),
            "optimal_spend": float(r["optimal_spend"]),
            "change_pct": float(r["change_pct"]),
        }
        # Extrapolation flag + recommended-spend CI (issue #105).
        if has_range:
            out["within_observed_range"] = bool(r["within_observed_range"])
            out["recommended_multiplier"] = float(r["recommended_multiplier"])
            mo = r.get("max_obs_multiplier")
            out["max_obs_multiplier"] = None if mo is None else float(mo)
            if "optimal_spend_p5" in t.columns:
                out["optimal_spend_p5"] = float(r["optimal_spend_p5"])
                out["optimal_spend_p95"] = float(r["optimal_spend_p95"])
        return out

    allocation = [_row(r) for _, r in t.iterrows()]
    plan: dict[str, Any] = {
        "total_budget": float(res.total_budget),
        "current_total": float(t["current_spend"].sum()),
        "expected_uplift": float(res.expected_uplift),
        "uplift_hdi": [float(res.uplift_hdi[0]), float(res.uplift_hdi[1])],
        "prob_positive_uplift": float(res.prob_positive_uplift),
        "expected_regret": float(res.expected_regret),
        "n_extrapolated": int(res.n_extrapolated),
        "n_draws": int(res.n_draws),
        "allocation": allocation,
        "notes": list(res.notes),
    }
    if deviation is not None:
        plan["deviation_cap"] = float(deviation)
    return plan


def default_reallocation(
    mmm: Any,
    *,
    deviation: float = DEFAULT_REALLOC_DEVIATION,
    max_draws: int = 150,
    random_seed: int | None = 42,
) -> dict[str, Any]:
    """A conservative, report-ready *default reallocation* of the current budget.

    This is the plan surfaced in a generated client report when the user has not
    run the Planner studio. It reallocates the **current total spend** (no budget
    change) across channels, but constrains every channel to within
    ``±deviation`` of its current spend — default **±20%** — so that:

    * **no channel is turned off** (the floor is ``1 - deviation`` of current
      spend, never zero), and
    * **no recommendation extrapolates** beyond the spend range the model has
      evidence for — the response curves are sampled on a multiplier grid that
      stays inside ``[1 - deviation, 1 + deviation]`` (plus the ``1.0`` anchor),
      so the greedy allocator only ever interpolates within observed support.

    Args:
        mmm: a fitted model exposing ``X_media_raw`` / ``channel_names`` /
            ``sample_channel_contributions`` (the :func:`compute_response_curves`
            contract).
        deviation: max fractional move per channel (0.20 ⇒ ±20%).
        max_draws: posterior draws for the curves and decision-uncertainty.
        random_seed: seed for reproducible curve sampling.

    Returns:
        The report-ready ``allocation_results`` dict (see
        :func:`_result_to_report_dict`), tagged with ``deviation_cap``.
    """
    deviation = float(deviation)
    if not (0.0 < deviation < 1.0):
        raise ValueError(f"deviation must be in (0, 1); got {deviation}.")
    lo, hi = 1.0 - deviation, 1.0 + deviation
    # A multiplier grid focused on the ±deviation band: endpoints included so the
    # bounds land on sampled points (no extrapolation), plus an interior point on
    # each side so the marginal-return interpolation has curvature to work with.
    grid = tuple(sorted({lo, (lo + 1.0) / 2.0, 1.0, (1.0 + hi) / 2.0, hi}))
    curves = compute_response_curves(
        mmm, multipliers=grid, max_draws=max_draws, random_seed=random_seed
    )
    res = optimize_budget(
        curves=curves,
        min_multiplier=lo,
        max_multiplier=hi,
        max_draws=max_draws,
        random_seed=random_seed,
    )
    return _result_to_report_dict(res, deviation=deviation)


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

    X = np.asarray(mmm.X_media_raw, dtype=float)
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
        # This geography's own observed per-period spend range (issue #121), so a
        # per-geo recommendation that scales a geo's spend past what that geo
        # actually ran is flagged, exactly like the national path (#105).
        n_geo_obs = int(mask.sum())
        obs_max_spend = X[mask].max(axis=0) if n_geo_obs else None
        # (G, D, C) -> (D, C, G)
        contributions = np.stack(per_mult_by_geo[g], axis=0).transpose(1, 2, 0)
        out[name] = ResponseCurves(
            channel_names=list(mmm.channel_names),
            multipliers=mults,
            base_spend=base_spend,
            contributions=contributions,
            obs_max_spend=obs_max_spend,
            n_obs=n_geo_obs,
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
    # Carry each geo's observed per-period spend range onto its arms (issue #121)
    # so the flattened optimizer flags a geo×channel arm scaled past that geo's
    # observed range. n_obs is kept PER ARM (its geo's period count) rather than a
    # single scalar so ragged panels — geos with different period counts — get the
    # right multiplier. Dropped only if any geo lacks the range (back-compat).
    arm_obs_max: list[float] = []
    arm_n_obs: list[int] = []
    has_range = True
    for g_idx, g in enumerate(geos):
        rc = geo_curves[g]
        for c, ch in enumerate(rc.channel_names):
            arm_names.append(f"{g_idx}{GEO_ARM_SEP}{ch}")
            base.append(float(rc.base_spend[c]))
            contribs.append(rc.contributions[:, c, :])  # (D, G_mult)
            if rc.obs_max_spend is None or rc.n_obs is None:
                has_range = False
            else:
                arm_obs_max.append(float(rc.obs_max_spend[c]))
                arm_n_obs.append(int(np.asarray(rc.n_obs)))
    contributions = np.stack(contribs, axis=1)  # (D, A, G_mult)
    return ResponseCurves(
        channel_names=arm_names,
        multipliers=mults,
        base_spend=np.asarray(base),
        contributions=contributions,
        obs_max_spend=np.asarray(arm_obs_max) if has_range else None,
        n_obs=np.asarray(arm_n_obs) if has_range else None,
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
