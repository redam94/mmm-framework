"""Switchback — time-randomized on/off design for national (no-geo) data.

The treatment is toggled on a randomized block schedule and the estimand is the
on-vs-off contrast. Two things dominate the power math, and both are about
TIME:

* **Carryover** — a block shorter than the channel's adstock memory smears the
  contrast (the "off" blocks still carry the "on" blocks' effect). The block
  length must be >= the carryover washout; with a fitted model
  :func:`planning.experiment_optimizer.cooldown_weeks` supplies it, otherwise a
  caller-provided ``carryover_weeks`` (default 2). An analysis *burn-in* — drop
  the first ``burn_in`` weeks of each block — removes the residual smear.
* **Autocorrelation** — KPI series are autocorrelated, so the i.i.d.
  two-sample SE understates the truth. The analytic MDE is inflated by the
  AR(1) design effect (:func:`planning.identification.ar1_design_effect`), and
  the A/A harness (block-aware by construction — the estimator only contrasts
  scheduled blocks) provides the calibrated/empirical numbers.

The *design* wraps :func:`planning.design.flighting_design` (block-randomized,
budget-neutral) — switchback IS a flighting design; this module contributes the
carryover-aware block sizing, the burn-in plan, and the honest power math. The
*analysis* reuses :func:`planning.simulation.national_onoff_estimator`
(registered under the ``switchback`` key), so the A/A·A/B leaderboard runs it
unchanged.

numpy/pandas only; the fitted model is optional everywhere.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..design import flighting_design, load_design_frame
from ..identification import ar1_design_effect

_Z_975 = 1.959963984540054
_MDE_FACTOR = 2.8  # z_.975 + z_.80 — the repo-wide convention (design.py)


def switchback_design(
    dataset_path: str,
    kpi: str,
    channel: str,
    *,
    duration: int = 12,
    amplitude_pct: float = 50.0,
    block_weeks: int | None = None,
    carryover_weeks: int = 2,
    mmm: Any = None,
    budget_neutral: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a carryover-aware switchback schedule + honest power read-out.

    ``block_weeks=None`` derives the block length from the channel's carryover:
    the fitted adstock washout when ``mmm`` is given (via ``cooldown_weeks``),
    else ``carryover_weeks``. The analysis plan drops a burn-in of
    ``ceil(carryover/2)`` weeks (capped at block-1) at each block boundary.
    """
    washout = int(carryover_weeks)
    washout_source = "assumed"
    if mmm is not None:
        try:
            from ..experiment_optimizer import cooldown_weeks

            cd = cooldown_weeks(mmm, channel)
            washout = int(cd.get("weeks", washout))
            washout_source = str(cd.get("source", "adstock"))
        except Exception:
            pass

    blk = int(block_weeks) if block_weeks else max(1, washout)
    design = flighting_design(
        dataset_path,
        kpi,
        channel,
        amplitude_pct=amplitude_pct,
        block_weeks=blk,
        duration=duration,
        budget_neutral=budget_neutral,
        seed=seed,
    )

    # burn-in: weeks dropped from the head of each block at analysis time
    burn_in = min(max(washout // 2, 1 if washout > blk else 0), max(blk - 1, 0))
    schedule = design.get("schedule") or []
    n_blocks = max(1, math.ceil(len(schedule) / blk)) if schedule else duration // blk
    n_switches = max(n_blocks - 1, 0)

    # AR(1)-honest analytic power on the on/off contrast. The estimator
    # contrasts LEVEL means across blocks, so the relevant autocorrelation is
    # that of the detrended level series — NOT first differences (differencing
    # flips smooth-series autocorrelation negative and the [0, rho_max] clip
    # would zero the design effect).
    frame = load_design_frame(dataset_path, kpi, channel)
    kpi_series = frame["kpi_national"].to_numpy(float)
    t = np.arange(len(kpi_series), dtype=float)
    if len(kpi_series) > 3:
        coef = np.polyfit(t, kpi_series, 1)
        resid_level = kpi_series - np.polyval(coef, t)
    else:  # pragma: no cover - degenerate series
        resid_level = kpi_series - float(np.mean(kpi_series))
    deff_info = ar1_design_effect(resid_level, blk)
    analysis_weeks = max(len(schedule) - n_blocks * burn_in, 2)
    sigma_y = float(np.std(resid_level)) if len(kpi_series) > 2 else float("nan")
    power = switchback_power(
        design,
        sigma_y=sigma_y,
        design_effect=float(deff_info.get("deff", 1.0)),
        analysis_weeks=analysis_weeks,
    )

    design.update(
        {
            "method": "switchback",
            "method_name": "Switchback (time-randomized)",
            "design_key": design.get("design_key", "national_flighting"),
            "block_weeks": blk,
            "carryover_weeks": washout,
            "carryover_source": washout_source,
            "burn_in_weeks": int(burn_in),
            "n_blocks": int(n_blocks),
            "n_switches": int(n_switches),
            "ar1": deff_info,
            "switchback_power": power,
        }
    )
    if blk < washout:
        design["carryover_warning"] = (
            f"block length {blk}w is shorter than the ~{washout}w carryover "
            "memory — the on/off contrast will be smeared; lengthen the blocks "
            "or increase the burn-in."
        )
    return design


def switchback_power(
    design: dict[str, Any],
    *,
    sigma_y: float,
    design_effect: float = 1.0,
    analysis_weeks: int | None = None,
) -> dict[str, Any]:
    """Analytic switchback MDE with the AR(1) design effect applied.

    The naive two-sample SE ``sigma * sqrt(1/n_on + 1/n_off)`` assumes i.i.d.
    weeks; the honest SE multiplies ``sigma`` by ``sqrt(deff)``. Both are
    reported so the inflation is visible.
    """
    schedule = design.get("schedule") or []
    mults = [s.get("multiplier", 1.0) for s in schedule]
    n_weeks = analysis_weeks if analysis_weeks is not None else len(mults)
    n_on = sum(1 for m in mults if m > 1.0)
    n_off = sum(1 for m in mults if m < 1.0)
    scale = (n_weeks / max(len(mults), 1)) if mults else 1.0
    n_on = max(1.0, n_on * scale)
    n_off = max(1.0, n_off * scale)

    se_iid = sigma_y * math.sqrt(1.0 / n_on + 1.0 / n_off)
    se_honest = se_iid * math.sqrt(max(design_effect, 1.0))
    out = {
        "sigma_y": float(sigma_y),
        "design_effect": float(design_effect),
        "n_on_weeks": float(n_on),
        "n_off_weeks": float(n_off),
        "se_iid": float(se_iid),
        "se_honest": float(se_honest),
        "mde_iid": float(_MDE_FACTOR * se_iid),
        "mde_honest": float(_MDE_FACTOR * se_honest),
        "mde_units": "KPI per week (on-vs-off step)",
    }
    spend_delta = design.get("weekly_spend_delta")
    if spend_delta:
        out["mde_roas_honest"] = float(out["mde_honest"] / abs(float(spend_delta)))
    return out


__all__ = ["switchback_design", "switchback_power"]
