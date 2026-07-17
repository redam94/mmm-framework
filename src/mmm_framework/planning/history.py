"""Per-run history metrics: the JSON snapshot each fit leaves behind so the
Performance page can plot cycle-over-cycle trajectories (ROI CI contraction,
budget-share migration, misallocation cost, portfolio marginal ROI) WITHOUT
ever unpickling old models.

``compute_run_metrics`` runs at fit time, kernel-side: it must touch only the
fitted model (no registry / DB access — that enrichment happens host-side in
``mmm_framework.api.history``) and never fail a fit (callers wrap it in
try/except). One ``compute_response_curves`` pass is shared by the ROI draws,
the budget optimization, and the EIG/EVOI grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .budget import compute_response_curves, optimize_budget
from .priority import compute_experiment_priorities

RUN_METRICS_SCHEMA_VERSION = 2


def compute_run_metrics(
    mmm: Any,
    *,
    max_draws: int = 200,
    n_outcomes: int = 48,
    random_seed: int | None = 42,
) -> dict[str, Any]:
    """JSON-safe per-run metrics snapshot (schema v2).

    Per channel: spend & share, ROI posterior (mean/sd/5–95% interval/width),
    marginal ROI at current spend, optimal-vs-current share gap, allocation
    instability, and the EIG/EVOI priority fields. Portfolio: total spend,
    portfolio marginal ROI, the misallocation proxy (median uplift of the
    optimal vs current allocation), v_current/EVPI, and the mean CI width
    (the contraction series). Schema v2 adds ``response_curves``: the compact
    per-channel saturation curve (spend grid × mean/5–95% contribution) the UI
    needs to show *why* average and marginal ROAS diverge.
    """
    curves = compute_response_curves(mmm, max_draws=max_draws, random_seed=random_seed)
    optimization = optimize_budget(curves=curves, random_seed=random_seed)
    grid, portfolio = compute_experiment_priorities(
        mmm,
        curves=curves,
        optimization=optimization,
        n_outcomes=n_outcomes,
        random_seed=random_seed,
    )
    by_name = {g.channel: g for g in grid}

    names = curves.channel_names
    base = curves.base_spend.astype(float)
    total_spend = float(base.sum())
    mults = curves.multipliers
    g1 = int(np.argmin(np.abs(mults - 1.0)))
    mean_curves = curves.mean_curves()  # (C, G)

    # Marginal ROI at current spend: forward difference to the next grid point
    # (falls back to the previous point when 1.0 is the top of the grid).
    if g1 + 1 < len(mults):
        lo_idx, hi_idx = g1, g1 + 1
    else:
        lo_idx, hi_idx = g1 - 1, g1
    d_mult = float(mults[hi_idx] - mults[lo_idx])

    opt = optimization.table.set_index("channel")

    channels: dict[str, Any] = {}
    marginal_total_num = 0.0
    marginal_total_den = 0.0
    ci_widths = []
    for c, name in enumerate(names):
        spend = float(base[c])
        roi_draws = curves.contributions[:, c, g1] / max(spend, 1e-12)
        p5, p95 = (float(np.percentile(roi_draws, q)) for q in (5, 95))
        ci_width = p95 - p5
        ci_widths.append(ci_width)

        d_contrib = float(mean_curves[c, hi_idx] - mean_curves[c, lo_idx])
        d_spend = spend * d_mult
        marginal_roi = d_contrib / d_spend if d_spend > 0 else 0.0
        marginal_total_num += d_contrib
        marginal_total_den += d_spend

        current_share = spend / max(total_spend, 1e-12)
        optimal_share = float(opt.loc[name, "optimal_share_pct"]) / 100.0
        g = by_name[name]
        channels[name] = {
            "spend": spend,
            "spend_share": current_share,
            "roi_mean": float(np.mean(roi_draws)),
            "roi_sd": float(np.std(roi_draws)),
            "roi_hdi_low": p5,
            "roi_hdi_high": p95,
            "ci_width": ci_width,
            "marginal_roi": marginal_roi,
            "current_share": current_share,
            "optimal_share": optimal_share,
            "share_gap": optimal_share - current_share,
            "allocation_instability": (
                float(opt.loc[name, "allocation_instability"]) / 100.0
            ),
            "eig": g.eig,
            "evoi": g.evoi,
            "sigma_exp": g.sigma_exp,
            "priority": g.priority,
            "quadrant": g.quadrant,
        }

    # Compact saturation curves: small (C × ~10 grid points), but enough for
    # the UI to draw each channel's response curve with its 5–95% band and
    # mark current spend — the visual case for marginal ≠ average ROAS.
    spend_grid = curves.spend_grid  # (C, G)
    p5_curves = np.percentile(curves.contributions, 5, axis=0)  # (C, G)
    p95_curves = np.percentile(curves.contributions, 95, axis=0)
    response_curves = {
        "multipliers": [float(m) for m in mults],
        "current_index": g1,
        "channels": {
            name: {
                "spend": [float(s) for s in spend_grid[c]],
                "mean": [float(v) for v in mean_curves[c]],
                "p5": [float(v) for v in p5_curves[c]],
                "p95": [float(v) for v in p95_curves[c]],
            }
            for c, name in enumerate(names)
        },
    }

    # Fit provenance so persisted ROI/CI history is not silently treated as
    # calibrated when it came from an approximate (MAP/Laplace/ADVI/Pathfinder)
    # fit. NUTS and SMC are exact (FitMethod.is_approximate).
    _fm = getattr(getattr(mmm, "model_config", None), "fit_method", None)
    _fit_method = getattr(_fm, "value", _fm)

    def _is_approx(val) -> bool | None:
        if val is None:
            return None
        try:
            from ..config.enums import FitMethod as _FM

            return _FM(str(val).lower()).is_approximate
        except ValueError:
            return str(val).lower() != "nuts"

    metrics = {
        "schema_version": RUN_METRICS_SCHEMA_VERSION,
        "n_draws": int(curves.contributions.shape[0]),
        "fit_method": str(_fit_method) if _fit_method is not None else None,
        "approximate": _is_approx(_fit_method),
        "response_curves": response_curves,
        "channels": channels,
        "portfolio": {
            "total_spend": total_spend,
            "portfolio_marginal_roi": (
                marginal_total_num / marginal_total_den
                if marginal_total_den > 0
                else 0.0
            ),
            # Misallocation proxy: what the current allocation leaves on the
            # table vs the optimum, per response window (median across draws).
            "expected_uplift": float(optimization.expected_uplift),
            "uplift_hdi": [float(x) for x in optimization.uplift_hdi],
            "prob_positive_uplift": float(optimization.prob_positive_uplift),
            "v_current": float(portfolio["v_current"]),
            "evpi": float(portfolio["evpi"]),
            "mean_ci_width": float(np.mean(ci_widths)) if ci_widths else 0.0,
        },
    }
    return metrics
