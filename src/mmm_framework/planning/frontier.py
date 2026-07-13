"""Efficient frontier & goal-seek on a fitted BayesianMMM (#139).

Budget optimizer v2 answers the questions a planner actually asks:

* **frontier / budget sweep** — how does optimized return scale as I add or cut
  budget? (:func:`budget_frontier`) — a spend-vs-return curve with credible bands
  and the marginal ROI (the frontier slope) at each budget.
* **goal-seek** — what total budget (and mix) hits my target KPI?
  (:func:`goal_seek`) — an inverse solve by bisection on the monotone frontier,
  with the probability the plan actually clears the target.

Both build on :mod:`mmm_framework.planning.budget`: the same posterior-sampled
:class:`ResponseCurves`, the same constrained SLSQP allocator (so risk
objectives, group / absolute constraints and the keep-on floor all apply here
too), and the same additive-model concavity that makes the frontier monotone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .budget import (
    ResponseCurves,
    _eval_allocation,
    _normalize_groups,
    _solve_allocation,
    compute_response_curves,
    objective_curves,
    objective_label,
)


@dataclass
class FrontierPoint:
    """One point on the efficient frontier: the optimized return at a budget."""

    total_budget: float
    expected_return: float  # optimized KPI (posterior mean of the plan)
    return_p5: float
    return_p95: float
    marginal_roi: float  # local frontier slope d(return)/d(budget)
    allocation: dict[str, float]


@dataclass
class FrontierResult:
    """The efficient frontier plus the current operating point."""

    objective: str
    objective_label: str
    channel_names: list[str]
    current_total: float
    current_return: float
    points: list[FrontierPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "objective_label": self.objective_label,
            "channels": list(self.channel_names),
            "current_total": float(self.current_total),
            "current_return": float(self.current_return),
            "points": [
                {
                    "total_budget": float(p.total_budget),
                    "expected_return": float(p.expected_return),
                    "return_p5": float(p.return_p5),
                    "return_p95": float(p.return_p95),
                    "marginal_roi": float(p.marginal_roi),
                    "allocation": {k: float(v) for k, v in p.allocation.items()},
                }
                for p in self.points
            ],
            "notes": list(self.notes),
        }


@dataclass
class GoalSeekResult:
    """The minimum budget (and mix) that reaches a target KPI contribution."""

    target_kpi: float
    objective: str
    objective_label: str
    channel_names: list[str]
    feasible: bool
    required_budget: float | None
    allocation: dict[str, float] | None
    expected_return: float | None
    prob_hit_target: float | None  # P(plan's KPI ≥ target) across draws
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_kpi": float(self.target_kpi),
            "objective": self.objective,
            "objective_label": self.objective_label,
            "channels": list(self.channel_names),
            "feasible": bool(self.feasible),
            "required_budget": (
                None if self.required_budget is None else float(self.required_budget)
            ),
            "allocation": (
                None
                if self.allocation is None
                else {k: float(v) for k, v in self.allocation.items()}
            ),
            "expected_return": (
                None if self.expected_return is None else float(self.expected_return)
            ),
            "prob_hit_target": (
                None if self.prob_hit_target is None else float(self.prob_hit_target)
            ),
            "notes": list(self.notes),
        }


def _prep(
    mmm: Any | None,
    curves: ResponseCurves | None,
    max_draws: int,
    random_seed: int | None,
) -> ResponseCurves:
    if curves is None:
        if mmm is None:
            raise ValueError("Provide either a fitted model or precomputed curves.")
        curves = compute_response_curves(
            mmm, max_draws=max_draws, random_seed=random_seed
        )
    return curves


def _bounds(
    curves: ResponseCurves,
    min_multiplier: float,
    max_multiplier: float,
    bounds: dict | None,
    abs_bounds: dict | None,
    min_channel_spend: float | dict | None,
    total_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    names, base = curves.channel_names, curves.base_spend
    grid_max = float(curves.multipliers.max())
    lo = np.array(
        [(bounds or {}).get(n, (min_multiplier, max_multiplier))[0] for n in names]
    )
    hi = np.array(
        [(bounds or {}).get(n, (min_multiplier, max_multiplier))[1] for n in names]
    )
    lo_spend = lo * base
    hi_spend = np.minimum(hi * base, base * grid_max)
    if abs_bounds:
        idx = {n: c for c, n in enumerate(names)}
        for n, (blo, bhi) in abs_bounds.items():
            if n in idx:
                lo_spend[idx[n]] = float(blo)
                hi_spend[idx[n]] = min(float(bhi), base[idx[n]] * grid_max)
    if min_channel_spend is not None:
        for c, n in enumerate(names):
            floor = (
                float(min_channel_spend.get(n, 0.0))
                if isinstance(min_channel_spend, dict)
                else float(min_channel_spend)
            )
            lo_spend[c] = max(lo_spend[c], min(floor, hi_spend[c]))
    return lo_spend, hi_spend


def budget_frontier(
    mmm: Any = None,
    *,
    curves: ResponseCurves | None = None,
    budget_min: float | None = None,
    budget_max: float | None = None,
    n_points: int = 12,
    objective: str = "mean",
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    bounds: dict[str, tuple[float, float]] | None = None,
    abs_bounds: dict[str, tuple[float, float]] | None = None,
    groups: list[dict] | None = None,
    min_channel_spend: float | dict[str, float] | None = None,
    value_per_kpi: float = 1.0,
    max_draws: int = 120,
    random_seed: int | None = 42,
) -> FrontierResult:
    """Sweep a budget range → the efficient frontier (optimized return vs budget).

    At each budget the constrained allocator maximizes the ``objective`` curve;
    the resulting plan is evaluated across all posterior draws for a return band,
    and the discrete slope gives the marginal ROI (the next-dollar return at that
    budget). By default the sweep runs from 40% to 200% of current spend.
    """
    curves = _prep(mmm, curves, max_draws, random_seed)
    names, base = curves.channel_names, curves.base_spend
    spend_grid = curves.spend_grid
    current_total = float(base.sum())
    lo = budget_min if budget_min is not None else 0.4 * current_total
    hi = budget_max if budget_max is not None else 2.0 * current_total
    lo = max(float(lo), 0.0)
    hi = max(float(hi), lo + 1e-9)
    budgets = np.linspace(lo, hi, max(int(n_points), 2))

    obj_curve = objective_curves(curves, objective)
    current_return = float(
        _eval_allocation(base.astype(float), curves.mean_curves(), spend_grid)
    )

    points: list[FrontierPoint] = []
    prev_ret: float | None = None
    prev_bud: float | None = None
    x0 = None
    for b in budgets:
        lo_spend, hi_spend = _bounds(
            curves,
            min_multiplier,
            max_multiplier,
            bounds,
            abs_bounds,
            min_channel_spend,
            float(b),
        )
        norm_groups = _normalize_groups(groups, names, float(b)) if groups else None
        alloc, _, _ = _solve_allocation(
            obj_curve,
            spend_grid,
            total_budget=float(b),
            lo_spend=lo_spend,
            hi_spend=hi_spend,
            groups=norm_groups,
            mode="fixed",
            value_per_kpi=value_per_kpi,
            x0=x0,
        )
        x0 = alloc
        # return distribution across draws under this plan
        rets = np.array(
            [
                _eval_allocation(alloc, curves.contributions[d], spend_grid)
                for d in range(curves.contributions.shape[0])
            ]
        )
        exp_ret = float(rets.mean())
        mroi = (
            (exp_ret - prev_ret) / (float(b) - prev_bud)
            if prev_ret is not None and float(b) > (prev_bud or 0)
            else float("nan")
        )
        points.append(
            FrontierPoint(
                total_budget=float(b),
                expected_return=exp_ret,
                return_p5=float(np.percentile(rets, 5)),
                return_p95=float(np.percentile(rets, 95)),
                marginal_roi=float(mroi),
                allocation={n: float(alloc[c]) for c, n in enumerate(names)},
            )
        )
        prev_ret, prev_bud = exp_ret, float(b)

    # backfill the first point's marginal ROI from the second (forward slope)
    if len(points) >= 2 and not np.isfinite(points[0].marginal_roi):
        points[0].marginal_roi = points[1].marginal_roi

    notes = [
        "Frontier optimizes the "
        f"{objective_label(objective)} at each budget; the band is the 5–95% "
        "posterior interval of the plan's return. Marginal ROI is the next-dollar "
        "return (the frontier slope) — invest while it stays above your breakeven."
    ]
    return FrontierResult(
        objective=objective,
        objective_label=objective_label(objective),
        channel_names=list(names),
        current_total=current_total,
        current_return=current_return,
        points=points,
        notes=notes,
    )


def goal_seek(
    mmm: Any = None,
    *,
    curves: ResponseCurves | None = None,
    target_kpi: float,
    objective: str = "mean",
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    bounds: dict[str, tuple[float, float]] | None = None,
    abs_bounds: dict[str, tuple[float, float]] | None = None,
    groups: list[dict] | None = None,
    min_channel_spend: float | dict[str, float] | None = None,
    value_per_kpi: float = 1.0,
    budget_max: float | None = None,
    tol: float = 1e-3,
    max_iter: int = 40,
    max_draws: int = 120,
    random_seed: int | None = 42,
) -> GoalSeekResult:
    """Inverse solve: the minimum budget (and its mix) whose optimized plan
    reaches ``target_kpi`` total media contribution.

    The frontier return is monotone increasing and concave in budget, so a
    bisection on the optimized return finds the required budget. Reports the
    plan and ``prob_hit_target`` — the posterior probability that the plan's
    contribution actually clears the target (the honest answer to "will this
    budget hit my number?").
    """
    curves = _prep(mmm, curves, max_draws, random_seed)
    names, base = curves.channel_names, curves.base_spend
    spend_grid = curves.spend_grid
    current_total = float(base.sum())
    obj_curve = objective_curves(curves, objective)

    def optimize_at(b: float) -> np.ndarray:
        lo_spend, hi_spend = _bounds(
            curves,
            min_multiplier,
            max_multiplier,
            bounds,
            abs_bounds,
            min_channel_spend,
            b,
        )
        norm_groups = _normalize_groups(groups, names, b) if groups else None
        alloc, _, _ = _solve_allocation(
            obj_curve,
            spend_grid,
            total_budget=b,
            lo_spend=lo_spend,
            hi_spend=hi_spend,
            groups=norm_groups,
            mode="fixed",
            value_per_kpi=value_per_kpi,
        )
        return alloc

    def ret_at(b: float) -> float:
        return _eval_allocation(optimize_at(b), obj_curve, spend_grid)

    hi = budget_max if budget_max is not None else 3.0 * current_total
    hi = float(max(hi, 1e-6))
    lo = 0.0
    # feasibility: even the max budget can't reach the target.
    if ret_at(hi) < target_kpi:
        return GoalSeekResult(
            target_kpi=float(target_kpi),
            objective=objective,
            objective_label=objective_label(objective),
            channel_names=list(names),
            feasible=False,
            required_budget=None,
            allocation=None,
            expected_return=None,
            prob_hit_target=None,
            notes=[
                f"Target of {target_kpi:g} is not reachable within the spend range "
                f"the model supports (max optimized return ≈ {ret_at(hi):g} at "
                f"budget {hi:g}). Raise budget_max or lower the target."
            ],
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if ret_at(mid) < target_kpi:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(current_total, 1.0):
            break
    b_req = hi
    alloc = optimize_at(b_req)
    rets = np.array(
        [
            _eval_allocation(alloc, curves.contributions[d], spend_grid)
            for d in range(curves.contributions.shape[0])
        ]
    )
    return GoalSeekResult(
        target_kpi=float(target_kpi),
        objective=objective,
        objective_label=objective_label(objective),
        channel_names=list(names),
        feasible=True,
        required_budget=float(b_req),
        allocation={n: float(alloc[c]) for c, n in enumerate(names)},
        expected_return=float(rets.mean()),
        prob_hit_target=float(np.mean(rets >= target_kpi)),
        notes=[
            f"Minimum budget to reach {target_kpi:g} is ≈ {b_req:g} "
            f"({100 * b_req / max(current_total, 1e-9):.0f}% of current spend). "
            "prob_hit_target is the posterior probability the plan actually clears "
            "the target — treat a low value as 'budget this, but the target is not "
            "assured'."
        ],
    )
