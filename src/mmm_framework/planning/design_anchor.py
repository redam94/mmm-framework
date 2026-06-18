"""Anchor a design's power analysis to the fitted model's expected effect.

The pure-data designer (``planning.design``) reports an MDE — the smallest ROAS
effect a test could detect — but says nothing about whether the effect we
actually EXPECT clears that bar. This module closes that gap using the model:

- ``model_anchored_effect`` perturbs ONLY the treated geo × test-window rows at
  the design's full intensity and reads the incremental KPI straight off the
  model's deterministic ``channel_contributions`` (paired draws, F1). It does
  NOT interpolate the global response curve — that scales every geo uniformly
  and mis-states a treated subset under heterogeneous saturation (F8/ANCHOR-2).
- ``powered_to_detect`` compares the design's MDE to that model-implied effect
  posterior → a powered / underpowered / overpowered / inconclusive verdict,
  a signed two-sided assurance, and the duration that would reach power.
- ``realized_sigma_exp_for_anchor`` turns the design's realized ROAS SE into the
  ``sigma_exp`` the EIG/EVOI engine expects, so the priority grid can use the
  ACTUAL design precision instead of a generic table.

numpy/pandas only — kernel-safe. Reuses the perturbation mechanics in
``opportunity_cost`` so the anchor and the opportunity cost see the same rows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .opportunity_cost import _resolve_treated_rows, build_experiment_media

# Two-sided z at alpha=0.05 (the design uses MDE_FACTOR = z_.975 + z_.80).
_Z_975 = 1.959963984540054
# MDE factor — local copy so this module need not import the pure-data design.py.
_FACTOR = 2.8
_EPS = 1e-9


def _phi(x: np.ndarray | float) -> np.ndarray | float:
    """Standard-normal CDF via erf (no scipy dependency)."""
    arr = np.asarray(x, dtype=float)
    out = 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    return float(out) if np.isscalar(x) else out


def model_anchored_effect(
    mmm: Any,
    design: dict,
    *,
    max_draws: int = 100,
    random_seed: int | None = 42,
    hdi_prob: float = 0.90,
    contrib_bau: np.ndarray | None = None,
    contrib_exp: np.ndarray | None = None,
) -> dict[str, Any]:
    """The model-implied effect the design would produce, as a ROAS posterior.

    Returns a JSON-safe dict with the incremental-ROAS draws (the estimand the
    experiment measures), its median/HDI, the expected incremental KPI, the
    average ROAS at current spend (context), and an extrapolation flag.

    ``contrib_bau``/``contrib_exp`` let a caller pass the (shared) posterior
    passes so the same BAU/experiment evaluation is not recomputed here and in
    ``compute_opportunity_cost``.
    """
    duration = int(design.get("duration", 8) or 8)
    treated_mask, treated_geo_codes, window_codes, dur_eff, _w = _resolve_treated_rows(
        mmm, design, duration=duration
    )
    x_exp, ch_idx, _n = build_experiment_media(
        mmm, design, treated_mask=treated_mask, window_codes=window_codes
    )
    x_bau = np.asarray(getattr(mmm, "X_media_raw"), dtype=np.float64)

    n_obs = x_bau.shape[0]
    if contrib_bau is None or np.asarray(contrib_bau).shape[1] != n_obs:
        contrib_bau = mmm.sample_channel_contributions(
            X_media=x_bau, max_draws=max_draws, random_seed=random_seed
        )
    if contrib_exp is None or np.asarray(contrib_exp).shape[1] != n_obs:
        contrib_exp = mmm.sample_channel_contributions(
            X_media=x_exp, max_draws=max_draws, random_seed=random_seed
        )
    delta_ch = contrib_exp[:, :, ch_idx] - contrib_bau[:, :, ch_idx]  # (D, n_obs)
    incr_kpi = delta_ch[:, treated_mask].sum(axis=1)  # (D,) window-only (F4)

    spend_bau = float(x_bau[treated_mask, ch_idx].sum())
    spend_exp = float(x_exp[treated_mask, ch_idx].sum())
    spend_delta = spend_exp - spend_bau  # signed; <0 for holdout
    if abs(spend_delta) <= _EPS:
        raise ValueError(
            "Design produces no spend change in the treated cells — cannot anchor "
            "an incremental ROAS."
        )
    # incr_kpi and spend_delta share a sign (lose KPI when you cut spend), so the
    # ratio is a positive ROAS for both holdout and scaling.
    incr_roas = incr_kpi / spend_delta  # (D,)

    # Average ROAS at current spend over the treated cells (context).
    roas_cur = contrib_bau[:, treated_mask, ch_idx].sum(axis=1) / max(spend_bau, _EPS)

    lo_q, hi_q = (100 * (1 - hdi_prob) / 2, 100 * (1 + hdi_prob) / 2)
    ch_hist_max = float(x_bau[:, ch_idx].max())
    return {
        "channel": design.get("channel"),
        "design_key": design.get("design_key"),
        "duration_effective": dur_eff,
        "n_draws": int(incr_roas.shape[0]),
        "spend_delta_modeled": spend_delta,
        "incremental_roas_draws": [float(v) for v in incr_roas],
        "incremental_roas_median": float(np.median(incr_roas)),
        "incremental_roas_hdi": [
            float(np.percentile(incr_roas, lo_q)),
            float(np.percentile(incr_roas, hi_q)),
        ],
        "expected_incremental_kpi_median": float(np.median(incr_kpi)),
        "expected_incremental_kpi_hdi": [
            float(np.percentile(incr_kpi, lo_q)),
            float(np.percentile(incr_kpi, hi_q)),
        ],
        "roas_at_current_median": float(np.median(roas_cur)),
        "roas_at_current_hdi": [
            float(np.percentile(roas_cur, lo_q)),
            float(np.percentile(roas_cur, hi_q)),
        ],
        "extrapolation_warning": bool(
            float(x_exp[treated_mask, ch_idx].max()) > ch_hist_max + _EPS
        ),
    }


def powered_to_detect(
    effect: dict,
    power_curve: list[dict] | None,
    duration: int,
    se_roas: float,
    *,
    overpowered_ratio: float = 2.0,
) -> dict[str, Any]:
    """Is the design powered to detect the model's expected effect?

    ``assurance`` is the SIGNED two-sided power averaged over the effect
    posterior — ``mean_d[Phi(eff_d/se - z) + Phi(-eff_d/se - z)]`` — so a null
    channel scores ~alpha rather than being rewarded for posterior width
    (ANCHOR-3). ``prob_detectable`` (the MDE indicator) leads the verdict.
    """
    draws = np.asarray(effect.get("incremental_roas_draws") or [], dtype=float)
    median = float(effect.get("incremental_roas_median", 0.0))
    hdi = effect.get("incremental_roas_hdi") or [median, median]
    mde_roas = float(_FACTOR * abs(se_roas)) if se_roas else float("inf")

    if draws.size and abs(se_roas) > _EPS:
        se = abs(se_roas)
        assurance = float(
            np.mean(_phi(draws / se - _Z_975) + _phi(-draws / se - _Z_975))
        )
        prob_detectable = float(np.mean(np.abs(draws) > mde_roas))
    else:
        assurance = float("nan")
        prob_detectable = float("nan")

    straddles_zero = float(hdi[0]) < 0.0 < float(hdi[1])
    if straddles_zero or not np.isfinite(prob_detectable):
        verdict = "inconclusive"
    elif prob_detectable >= 0.8:
        if np.isfinite(mde_roas) and mde_roas <= abs(median) / overpowered_ratio:
            verdict = "overpowered"
        else:
            verdict = "powered"
    else:
        verdict = "underpowered"

    recommended_duration: int | None = None
    if power_curve:
        target = abs(median)
        feasible = [
            int(p["duration"])
            for p in sorted(power_curve, key=lambda p: p["duration"])
            if float(p.get("mde_roas", float("inf"))) <= target
        ]
        recommended_duration = feasible[0] if feasible else None

    return {
        "verdict": verdict,
        "assurance": assurance if np.isfinite(assurance) else None,
        "prob_detectable": prob_detectable if np.isfinite(prob_detectable) else None,
        "mde_roas": mde_roas if np.isfinite(mde_roas) else None,
        "incremental_roas_median": median,
        "duration": int(duration),
        "recommended_duration": recommended_duration,
    }


def realized_sigma_exp_for_anchor(
    incremental_roas_draws: np.ndarray,
    se_roas: float,
    *,
    roi_floor: float = 0.05,
    floor: float = 1e-6,
    rel_lower: float = 0.05,
    rel_upper: float = 50.0,
) -> tuple[float, np.ndarray]:
    """(sigma_exp, incremental_roas_draws) for the EIG/EVOI loopback.

    ``sigma_exp`` is the design's realized ROAS SE, clamped so a flat-placebo
    ``se_roas -> 0`` can't return an absurd EIG and an exploded SE can't silently
    zero a strong design's EIG (ANCHOR-4). It pairs with
    ``incremental_roas_draws`` — the SAME estimand the experiment measures, not
    the average-ROAS posterior the priority grid uses by default (ANCHOR-1).
    """
    draws = np.asarray(incremental_roas_draws, dtype=float)
    scale = max(abs(float(np.median(draws))) if draws.size else 0.0, roi_floor)
    lower = max(floor, rel_lower * scale)
    upper = rel_upper * scale
    sigma_exp = float(np.clip(abs(float(se_roas)), lower, upper))
    return sigma_exp, draws
