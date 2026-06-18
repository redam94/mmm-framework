"""Experiment-design recommendations from a fitted MMM.

Ranks channels by how much an experiment would improve the NEXT decision, not
just by posterior width: a channel matters when (a) real money rides on it,
(b) its ROAS is uncertain, and (c) that uncertainty actually moves the optimal
allocation across posterior draws. The output includes a concrete design
(type, magnitude, duration from the channel's adstock window) and the exact
``ExperimentMeasurement`` calibration snippet that folds the result into the
next fit — closing the MMM → experiment → calibrated-MMM loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .budget import (
    BudgetOptimizationResult,
    ResponseCurves,
    compute_response_curves,
    optimize_budget,
)


def _adstock_l_max(mmm: Any, channel: str, default: int = 8) -> int:
    try:
        cfg = next(
            m
            for m in mmm.mff_config.media_channels
            if getattr(m, "name", None) == channel
        )
        return int(getattr(getattr(cfg, "adstock", None), "l_max", default))
    except Exception:
        return default


def recommend_experiments(
    mmm: Any,
    *,
    curves: ResponseCurves | None = None,
    optimization: BudgetOptimizationResult | None = None,
    top_k: int = 3,
    max_draws: int = 200,
    random_seed: int | None = None,
    method: str = "eig_evoi",
    evidence: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Rank channels by experiment value and propose concrete designs.

    Returns ``(table, designs)``: a per-channel scoring table (all channels,
    sorted by priority) and a list of design dicts for the ``top_k`` channels.

    method="eig_evoi" (default): priority is the normalized geometric mean of
    EIG (what the experiment teaches) and EVOI (what that learning is worth to
    the budget decision) — see ``planning.priority``. The table gains eig /
    evoi / quadrant columns and designs carry the priority snapshot.

    method="heuristic": the transparent legacy score
        spend_share × roas_cv × (1 + allocation_instability)
    — money at stake, times relative ROAS uncertainty, amplified when the
    uncertainty visibly destabilizes the optimal allocation.
    """
    if method not in ("eig_evoi", "heuristic"):
        raise ValueError(f"Unknown method '{method}'. Valid: eig_evoi, heuristic")
    if curves is None:
        curves = compute_response_curves(
            mmm, max_draws=max_draws, random_seed=random_seed
        )
    if optimization is None:
        optimization = optimize_budget(
            curves=curves, max_draws=max_draws, random_seed=random_seed
        )

    names = curves.channel_names
    base = curves.base_spend
    total_spend = float(base.sum())
    g1 = int(np.argmin(np.abs(curves.multipliers - 1.0)))
    contrib_at_current = curves.contributions[:, :, g1]  # (D, C)

    opt = optimization.table.set_index("channel")

    priority_grid: dict[str, Any] = {}
    if method == "eig_evoi":
        from .priority import compute_experiment_priorities

        grid, _portfolio = compute_experiment_priorities(
            mmm,
            curves=curves,
            optimization=optimization,
            evidence=evidence,
            random_seed=random_seed,
        )
        priority_grid = {g.channel: g for g in grid}

    rows = []
    for c, name in enumerate(names):
        spend = float(base[c])
        roas_draws = contrib_at_current[:, c] / max(spend, 1e-12)
        roas_med = float(np.median(roas_draws))
        roas_sd = float(np.std(roas_draws))
        p5, p95 = (float(np.percentile(roas_draws, q)) for q in (5, 95))
        cv = roas_sd / max(abs(roas_med), 1e-9)
        instability = float(opt.loc[name, "allocation_instability"]) / 100.0
        spend_share = spend / max(total_spend, 1e-12)
        row = {
            "channel": name,
            "spend_share_pct": 100 * spend_share,
            "roas_median": roas_med,
            "roas_p5": p5,
            "roas_p95": p95,
            "roas_cv": cv,
            "prob_roas_below_1": float(np.mean(roas_draws < 1.0)),
            "allocation_instability_pct": 100 * instability,
        }
        if method == "eig_evoi":
            g = priority_grid[name]
            row.update(
                {
                    "eig": g.eig,
                    "evoi": g.evoi,
                    "evpi_share": g.evpi_share,
                    "quadrant": g.quadrant,
                    "priority": g.priority,
                }
            )
        else:
            row["priority"] = spend_share * cv * (1.0 + instability)
        rows.append(row)

    table = (
        pd.DataFrame(rows)
        .sort_values("priority", ascending=False)
        .reset_index(drop=True)
    )

    has_geo = bool(getattr(mmm, "has_geo", False))
    designs: list[dict] = []
    for _, row in table.head(top_k).iterrows():
        name = row["channel"]
        l_max = _adstock_l_max(mmm, name)
        duration = l_max + 4  # carryover window + measurement periods
        # An informative experiment should at least halve the posterior sd
        roas_sd = (row["roas_p95"] - row["roas_p5"]) / 3.29  # ~sd from 90% interval
        target_se = roas_sd / 2.0
        design_key = "geo_holdout" if has_geo else "national_pulse"
        design_type = (
            "geo holdout / geo lift test"
            if has_geo
            else "national spend pulse (sustained +/-20% or dark period)"
        )
        why = (
            f"{row['spend_share_pct']:.0f}% of spend with ROAS 90% CI "
            f"[{row['roas_p5']:.2f}, {row['roas_p95']:.2f}] "
            f"(median {row['roas_median']:.2f}); optimal share swings "
            f"{row['allocation_instability_pct']:.0f} pts across posterior draws."
        )
        if method == "eig_evoi":
            g = priority_grid[name]
            # design-grounded precision wins when tighter than the heuristic
            target_se = min(target_se, g.sigma_exp)
            why += (
                f" EIG {g.eig:.2f} nats; EVOI {g.evoi:,.0f} KPI units "
                f"({g.evpi_share:.0%} of EVPI); quadrant: {g.quadrant}."
            )
        designs.append(
            {
                "channel": name,
                "priority": float(row["priority"]),
                "priority_method": method,
                **(
                    {
                        "eig": float(row["eig"]),
                        "evoi": float(row["evoi"]),
                        "quadrant": str(row["quadrant"]),
                        "sigma_exp": float(priority_grid[name].sigma_exp),
                    }
                    if method == "eig_evoi"
                    else {}
                ),
                "why": why,
                "design_type": design_type,
                "design_key": design_key,
                "min_duration_periods": int(duration),
                "duration_rationale": (
                    f"adstock l_max={l_max} carryover window + 4 measurement periods"
                ),
                "target_se": float(target_se),
                "target_se_rationale": (
                    "an experiment SE at or below half the model's ROAS posterior "
                    "sd meaningfully shrinks the posterior when calibrated in"
                ),
                "calibration_snippet": (
                    "from mmm_framework.calibration.likelihood import "
                    "ExperimentMeasurement, ExperimentEstimand\n"
                    "exp = ExperimentMeasurement(\n"
                    f"    channel={name!r},\n"
                    "    test_period=(<start_date>, <end_date>),\n"
                    "    value=<measured ROAS>,\n"
                    "    se=<standard error>,\n"
                    "    estimand=ExperimentEstimand.ROAS,\n"
                    ")\n"
                    "# then refit with the experiment folded into the likelihood:\n"
                    "# mmm.add_experiment_calibration([exp]) before .fit()"
                ),
            }
        )

    return table, designs
