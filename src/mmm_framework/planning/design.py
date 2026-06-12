"""Experiment design engine: turn "test this channel" into a runnable design.

Three design families, chosen by the data's granularity:

- **Randomized geo lift** (DMA/geo panels): model-structured matched pairs —
  each geo's KPI is residualized on the MMM's own baseline structure (trend +
  yearly seasonality + the channel's spend response) and pairs are matched on
  RESIDUAL co-movement plus a standardized covariate distance on the model's
  terms, solved as an optimal (blossom) matching. Treatment is RANDOMIZED
  within pairs (the randomization is what buys unconfoundedness). Power is a
  DiD analysis whose level is CALIBRATED against a placebo simulation (the
  estimator run over every historical window), with a covariate balance table.
- **Matched-market DiD** (pseudo-experimental): the same matching + power
  machinery without randomization, for when treatment geos are dictated by
  the business — the placebo distribution calibrates how big a "lift" the
  pre-period produces by chance, and the caveat is explicit.
- **Randomized flighting** (national data, or any single series): a
  budget-neutral, block-randomized on/off spend schedule that manufactures the
  exogenous variance a collinear history never provides — with the induced
  identification gain and an on/off-contrast power estimate.

All functions are pandas/numpy only (kernel-safe) and work straight off an
MFF csv — no fitted model required, so the endpoint stays fast.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# 80% power at two-sided alpha=0.05: z_{0.975} + z_{0.80} = 1.96 + 0.84
MDE_FACTOR = 2.8
DEFAULT_DURATIONS = (4, 6, 8, 10, 12, 16)


# ── Data access ───────────────────────────────────────────────────────────────


def load_design_frame(dataset_path: str, kpi: str, channel: str) -> dict[str, Any]:
    """Pivot an MFF csv into the matrices the designs need.

    Returns ``{kpi_wide, spend_wide, kpi_national, spend_national, periods,
    geos}`` where the ``*_wide`` frames are weeks × geos (None for national
    data: no Geography column, or a single geography).
    """
    df = pd.read_csv(dataset_path)
    if "VariableName" not in df.columns or "VariableValue" not in df.columns:
        raise ValueError("Not an MFF file: VariableName/VariableValue missing")
    df["Period"] = pd.to_datetime(df["Period"])

    has_geo = (
        "Geography" in df.columns
        and df["Geography"].notna().any()
        and df.loc[df["VariableName"] == kpi, "Geography"].nunique() > 1
    )

    def _series(name: str) -> pd.Series:
        sub = df[df["VariableName"] == name]
        return sub.groupby("Period")["VariableValue"].sum().sort_index()

    def _wide(name: str) -> pd.DataFrame:
        sub = df[df["VariableName"] == name]
        return (
            sub.pivot_table(
                index="Period",
                columns="Geography",
                values="VariableValue",
                aggfunc="sum",
            )
            .sort_index()
            .dropna(axis=1, how="all")
        )

    kpi_nat = _series(kpi)
    if kpi_nat.empty:
        raise ValueError(f"KPI '{kpi}' not found in the dataset")
    spend_nat = _series(channel)
    if spend_nat.empty:
        raise ValueError(f"Channel '{channel}' not found in the dataset")

    kpi_wide = _wide(kpi) if has_geo else None
    spend_wide = _wide(channel) if has_geo else None
    return {
        "kpi_wide": kpi_wide,
        "spend_wide": spend_wide,
        "kpi_national": kpi_nat,
        "spend_national": spend_nat,
        "periods": list(kpi_nat.index),
        "geos": list(kpi_wide.columns) if kpi_wide is not None else [],
    }


# ── Matching (model-structured) ───────────────────────────────────────────────
#
# Raw KPI correlation is a TRAP for geo matching: it is dominated by the trend
# and seasonality every geo shares, so two geos can look like a perfect pair
# while their idiosyncratic movements — the only thing a DiD's noise consists
# of — are unrelated. We therefore residualize each geo's KPI on the same
# baseline structure the MMM itself fits (linear trend + yearly Fourier
# harmonics + the tested channel's spend response, geo-by-geo OLS) and match
# on what's left:
#
#   distance(i, j) = (1 − corr(resid_i, resid_j))            # DiD noise core
#                  + 0.5 · ‖z_i − z_j‖ / √k                  # comparability
#
# where z is the standardized per-geo feature vector on the model's terms:
# log KPI level, trend slope, seasonal amplitude, residual volatility, and the
# channel's spend share. Pairing is solved as a minimum-weight perfect
# matching (blossom algorithm) — globally optimal, not greedy.

_YEARLY_PERIOD = 52.0
_FOURIER_ORDER = 2


def _baseline_design_matrix(n: int) -> np.ndarray:
    """Per-geo baseline regressors mirroring the MMM's structure: intercept,
    linear trend, yearly Fourier harmonics."""
    t = np.arange(n, dtype=float)
    cols = [np.ones(n), t / max(n - 1, 1)]
    for k in range(1, _FOURIER_ORDER + 1):
        cols.append(np.sin(2 * np.pi * k * t / _YEARLY_PERIOD))
        cols.append(np.cos(2 * np.pi * k * t / _YEARLY_PERIOD))
    return np.column_stack(cols)


def residualize_geo_panel(
    kpi_wide: pd.DataFrame, spend_wide: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Remove the model-explained structure from each geo's KPI.

    Geo-by-geo OLS of KPI on [intercept, trend, yearly Fourier(2), channel
    spend (standardized, when available)] — the deterministic skeleton of the
    MMM applied per geo. Returns ``{"residuals": DataFrame weeks×geos,
    "features": DataFrame geos×feature}`` where features live on the model's
    terms: level (log mean), trend_slope, seasonal_amplitude (Fourier
    coefficient norm), residual_sd, spend_share.
    """
    n = len(kpi_wide)
    x_base = _baseline_design_matrix(n)
    total_spend = float(spend_wide.sum().sum()) if spend_wide is not None else 0.0
    residuals: dict[str, np.ndarray] = {}
    feats: dict[str, dict[str, float]] = {}
    for g in kpi_wide.columns:
        y = kpi_wide[g].to_numpy(float)
        x = x_base
        geo_spend = 0.0
        if spend_wide is not None and g in spend_wide.columns:
            s = spend_wide[g].reindex(kpi_wide.index).fillna(0.0).to_numpy(float)
            geo_spend = float(s.sum())
            if s.std() > 0:
                x = np.column_stack([x_base, (s - s.mean()) / s.std()])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        residuals[g] = resid
        feats[g] = {
            "level": float(np.log(max(y.mean(), 1e-9))),
            "trend_slope": float(beta[1]),
            "seasonal_amplitude": float(
                np.sqrt(np.sum(beta[2 : 2 + 2 * _FOURIER_ORDER] ** 2))
            ),
            "residual_sd": float(resid.std()),
            "spend_share": (geo_spend / total_spend if total_spend > 0 else 0.0),
        }
    return {
        "residuals": pd.DataFrame(residuals, index=kpi_wide.index),
        "features": pd.DataFrame(feats).T,
    }


def _pair_distance(
    kpi_wide: pd.DataFrame, spend_wide: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """(distance matrix, residualization) per the module-level formula."""
    rz = residualize_geo_panel(kpi_wide, spend_wide)
    rcorr = rz["residuals"].corr()
    feats = rz["features"]
    z = (feats - feats.mean()) / feats.std().replace(0, 1.0)
    zv = z.to_numpy(float)
    k = zv.shape[1]
    feat_dist = np.sqrt(((zv[:, None, :] - zv[None, :, :]) ** 2).sum(-1) / k)
    d = (1.0 - rcorr) + 0.5 * pd.DataFrame(
        feat_dist, index=rcorr.index, columns=rcorr.columns
    )
    np.fill_diagonal(d.values, np.inf)
    return d, rz


def _optimal_pairing(d: pd.DataFrame) -> list[tuple[str, str]]:
    """Minimum-weight perfect matching (blossom) over the distance matrix;
    greedy fallback if networkx is unavailable. With an odd geo count the
    worst-matched geo sits out."""
    cols = list(d.columns)
    try:
        import networkx as nx

        g = nx.Graph()
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                w = float(d.loc[a, b])
                if np.isfinite(w):
                    g.add_edge(a, b, weight=w)
        mate = nx.min_weight_matching(g)
        return [(str(a), str(b)) for a, b in mate]
    except Exception:  # noqa: BLE001 - greedy fallback
        remaining = set(cols)
        pairs: list[tuple[str, str]] = []
        while len(remaining) >= 2:
            sub = d.loc[sorted(remaining), sorted(remaining)]
            idx = np.unravel_index(np.argmin(sub.values), sub.shape)
            a, b = sub.index[idx[0]], sub.columns[idx[1]]
            pairs.append((str(a), str(b)))
            remaining -= {a, b}
        return pairs


def matched_pairs(
    kpi_wide: pd.DataFrame,
    n_pairs: int | None = None,
    spend_wide: pd.DataFrame | None = None,
) -> list[dict]:
    """Model-structured matched pairs, best-match first.

    Each pair carries ``correlation`` (raw KPI — what naive matching would
    have scored), ``residual_correlation`` (co-movement AFTER removing trend/
    seasonality/spend — what the DiD's precision actually depends on),
    ``size_ratio``, and the matching ``distance``.
    """
    d, rz = _pair_distance(kpi_wide, spend_wide)
    raw_corr = kpi_wide.corr()
    rcorr = rz["residuals"].corr()
    means = kpi_wide.mean()
    pairs: list[dict] = []
    for a, b in _optimal_pairing(d):
        big, small = (a, b) if means[a] >= means[b] else (b, a)
        pairs.append(
            {
                "geo_a": a,
                "geo_b": b,
                "correlation": float(raw_corr.loc[a, b]),
                "residual_correlation": float(rcorr.loc[a, b]),
                "size_ratio": float(means[big] / max(means[small], 1e-9)),
                "distance": float(d.loc[a, b]),
            }
        )
    pairs.sort(key=lambda p: p["distance"])
    if n_pairs is not None:
        pairs = pairs[: max(1, n_pairs)]
    return pairs


def _balance_table(
    features: pd.DataFrame, treatment: list[str], control: list[str]
) -> list[dict[str, float | str]]:
    """Covariate balance on the model's terms: treated vs control means and
    the absolute standardized difference per feature (|Δ|/sd; < 0.25 is the
    conventional 'balanced' bar)."""
    rows: list[dict[str, float | str]] = []
    for col in features.columns:
        t_mean = float(features.loc[treatment, col].mean())
        c_mean = float(features.loc[control, col].mean())
        sd = float(features[col].std()) or 1.0
        rows.append(
            {
                "feature": str(col),
                "treatment_mean": t_mean,
                "control_mean": c_mean,
                "abs_std_diff": abs(t_mean - c_mean) / sd,
            }
        )
    return rows


# ── Power math ────────────────────────────────────────────────────────────────


def _diff_noise_sd(series: pd.Series | np.ndarray) -> float:
    """Noise sd of a (difference) series, robust to shared slow trend and
    seasonality: sd of first differences / sqrt(2)."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(np.std(np.diff(x)) / math.sqrt(2.0))


def _did_se_total_kpi(sigma_d: float, t_test: int, t_pre: int) -> float:
    """SE of the TOTAL incremental KPI over the test window for a DiD on the
    pooled treatment-minus-control series: the estimate is
    T·(mean_post(D) − mean_pre(D)), so
    Var = T²·σ_D²·(1/T + 1/T_pre) = σ_D²·T·(1 + T/T_pre)."""
    return sigma_d * math.sqrt(t_test * (1.0 + t_test / max(t_pre, 1)))


def _placebo_did(d_series: np.ndarray, t_test: int) -> dict[str, Any]:
    """Sliding-window placebo: how big a 'lift' does the PRE period produce by
    chance? Each window of length t_test is scored as a DiD against the rest
    of the series — the spread calibrates the falsification bar."""
    x = np.asarray(d_series, dtype=float)
    n = x.size
    if n < 2 * t_test:
        return {"n_windows": 0, "sd": None, "p95_abs": None}
    estimates = []
    for start in range(0, n - t_test + 1):
        window = x[start : start + t_test]
        rest = np.concatenate([x[:start], x[start + t_test :]])
        estimates.append(t_test * (window.mean() - rest.mean()))
    est = np.array(estimates)
    return {
        "n_windows": int(est.size),
        "sd": float(est.std()),
        "p95_abs": float(np.percentile(np.abs(est), 95)),
    }


# ── Geo lift / matched-market DiD ─────────────────────────────────────────────


def geo_lift_design(
    dataset_path: str,
    kpi: str,
    channel: str,
    *,
    design: str = "holdout",  # "holdout" (go dark) | "scaling" (spend lift)
    intensity_pct: float = 50.0,  # spend change in TREATED geos (scaling only)
    n_pairs: int | None = None,
    duration: int = 8,
    durations: tuple[int, ...] = DEFAULT_DURATIONS,
    randomize: bool = True,  # False -> matched-market DiD (pseudo-experimental)
    seed: int = 42,
) -> dict[str, Any]:
    """Design a geo lift study (or a matched-market DiD when randomize=False).

    Pipeline — robust by construction, on the model's own view of history:

    1. Residualize each geo's KPI on the MMM's baseline structure (trend +
       yearly seasonality + the channel's spend response).
    2. Match on residual co-movement + standardized model-feature distance;
       solve the pairing globally (minimum-weight perfect matching).
    3. Randomize treatment within each pair (or largest-geo-treated for the
       observational matched-market variant).
    4. Power: the analytic DiD curve is CALIBRATED against a placebo
       simulation — the actual estimator run over historical windows — so the
       reported SE/MDE reflects how this design behaves on this data, not
       just a formula. Covariate balance on the model's features is reported.
    """
    frame = load_design_frame(dataset_path, kpi, channel)
    kpi_wide, spend_wide = frame["kpi_wide"], frame["spend_wide"]
    if kpi_wide is None or kpi_wide.shape[1] < 4:
        raise ValueError(
            "Geo designs need a geo-level panel with at least 4 geographies — "
            f"found {0 if kpi_wide is None else kpi_wide.shape[1]}. Use the "
            "randomized flighting design for national data."
        )
    if design not in ("holdout", "scaling"):
        raise ValueError(f"Unknown geo design '{design}'. Valid: holdout, scaling")

    pairs = matched_pairs(kpi_wide, n_pairs, spend_wide=spend_wide)
    rz = residualize_geo_panel(kpi_wide, spend_wide)
    rng = np.random.default_rng(seed)

    assignment: list[dict] = []
    for p in pairs:
        a, b = p["geo_a"], p["geo_b"]
        if randomize:
            treat, control = (a, b) if rng.random() < 0.5 else (b, a)
        else:
            # observational matched-market: the bigger geo runs the change
            means = kpi_wide[[a, b]].mean()
            treat, control = (a, b) if means[a] >= means[b] else (b, a)
        assignment.append(
            {
                "treatment": treat,
                "control": control,
                "correlation": p["correlation"],
                "residual_correlation": p["residual_correlation"],
                "size_ratio": p["size_ratio"],
            }
        )

    treatment_geos = [p["treatment"] for p in assignment]
    control_geos = [p["control"] for p in assignment]

    # Pooled treatment-minus-control series: RAW for the placebo simulation
    # (it must exercise the estimator exactly as it will run), RESIDUALIZED
    # for the analytic noise floor (structure the matching already removed
    # must not inflate sigma).
    diff_raw = sum(
        kpi_wide[p["treatment"]].to_numpy(float)
        - kpi_wide[p["control"]].to_numpy(float)
        for p in assignment
    ) / len(assignment)
    diff_resid = sum(
        rz["residuals"][p["treatment"]].to_numpy(float)
        - rz["residuals"][p["control"]].to_numpy(float)
        for p in assignment
    ) / len(assignment)
    sigma_d = float(np.std(diff_resid))
    t_pre = int(len(diff_raw))

    # Spend delta per week in the treated cell
    spend_w = spend_wide if spend_wide is not None else None
    treated_weekly_spend = (
        float(spend_w[treatment_geos].sum(axis=1).mean())
        if spend_w is not None
        else 0.0
    )
    intensity = -100.0 if design == "holdout" else float(intensity_pct)
    weekly_spend_delta = abs(treated_weekly_spend * intensity / 100.0)
    if weekly_spend_delta <= 0:
        raise ValueError(
            f"Channel '{channel}' has no spend in the treated geos — pick a "
            "different channel or design."
        )

    # Simulation calibration: the placebo distribution IS the design-based SE
    # of the estimator on this data (every historical window of the chosen
    # length, scored exactly as the experiment will be). When enough windows
    # exist, scale the smooth analytic curve so it agrees with the simulation
    # at the chosen duration — honest level, smooth shape.
    placebo = _placebo_did(diff_raw, duration)
    analytic_at_chosen = _did_se_total_kpi(sigma_d, int(duration), t_pre)
    se_source = "analytic"
    calibration_factor = 1.0
    if placebo["n_windows"] >= 12 and placebo["sd"] and analytic_at_chosen > 0:
        calibration_factor = float(placebo["sd"]) / analytic_at_chosen
        se_source = "placebo_calibrated"

    # the pooled diff averages pairs, so the KPI-scale SE refers to the average
    # pair; total-cell lift scales by n_pairs in BOTH numerator and spend, so
    # ROAS-scale SE uses the per-pair spend share
    n_p = len(assignment)
    power_curve = []
    for t in sorted(set([*durations, duration])):
        se_total_avg_pair = calibration_factor * _did_se_total_kpi(
            sigma_d, int(t), t_pre
        )
        spend_delta_avg_pair = weekly_spend_delta * t / n_p
        se_roas = se_total_avg_pair / max(spend_delta_avg_pair, 1e-12)
        power_curve.append(
            {
                "duration": int(t),
                "se_roas": float(se_roas),
                "mde_roas": float(MDE_FACTOR * se_roas),
            }
        )
    chosen = next(p for p in power_curve if p["duration"] == duration)

    min_corr = min(p["correlation"] for p in assignment)
    min_resid_corr = min(p["residual_correlation"] for p in assignment)
    balance = _balance_table(rz["features"], treatment_geos, control_geos)
    max_imbalance = max((r["abs_std_diff"] for r in balance), default=0.0)

    method = (
        "randomized geo lift" if randomize else "matched-market DiD (observational)"
    )
    analysis_plan = (
        f"Estimator: difference-in-differences on the pooled treatment-minus-"
        f"control KPI series, intention-to-treat. Matching: geos residualized "
        f"on the model's baseline structure (trend, yearly seasonality, "
        f"{channel} spend), paired by residual co-movement + covariate "
        f"distance, solved as an optimal matching. Pre-register: {n_p} matched "
        f"pairs, {duration}-week test window, spend {'-100%' if design == 'holdout' else f'{intensity:+.0f}%'} "
        f"in treated geos, two-sided alpha=0.05. Power is "
        f"{'simulation-calibrated (placebo windows over history)' if se_source == 'placebo_calibrated' else 'analytic'}. "
        f"Convert the total incremental KPI to ROAS by dividing by the "
        f"realized spend change; calibrate into the next fit with "
        f"estimand='roas'. Falsification: the placebo distribution below is "
        f"the lift the pre-period produces by chance — a result inside its "
        f"95% band is noise."
        + (
            ""
            if randomize
            else " CAVEAT: assignment is NOT randomized; parallel-trends is an "
            "assumption, not a design guarantee — report the placebo band "
            "prominently and treat the readout as pseudo-experimental."
        )
    )

    return {
        "design_key": "geo_lift" if randomize else "matched_market_did",
        "design_type": f"{method} — {design}",
        "channel": channel,
        "kpi": kpi,
        "randomized": bool(randomize),
        "n_pairs": n_p,
        "assignment": assignment,
        "treatment_geos": treatment_geos,
        "control_geos": control_geos,
        "intensity_pct": intensity,
        "duration": int(duration),
        "weekly_spend_delta": float(weekly_spend_delta),
        "se_roas": chosen["se_roas"],
        "mde_roas": chosen["mde_roas"],
        "se_source": se_source,
        "power_curve": power_curve,
        "placebo": placebo,
        "balance": balance,
        "diagnostics": {
            "sigma_pair_diff": float(sigma_d),
            "pre_period_weeks": t_pre,
            "matching": (
                "model-structured residuals (trend + yearly seasonality + "
                "channel spend removed), optimal min-weight pairing"
            ),
            "calibration_factor": float(calibration_factor),
            "min_pair_correlation": float(min_corr),
            "min_residual_correlation": float(min_resid_corr),
            "max_balance_abs_std_diff": float(max_imbalance),
            # residual co-movement is what the DiD's noise depends on; raw
            # correlation can look great on shared seasonality alone
            "parallel_trends_warning": bool(min_resid_corr < 0.2),
        },
        "analysis_plan": analysis_plan,
        "seed": int(seed),
    }


# ── Randomized flighting (national) ───────────────────────────────────────────


def flighting_design(
    dataset_path: str,
    kpi: str,
    channel: str,
    *,
    amplitude_pct: float = 50.0,
    block_weeks: int = 2,
    duration: int = 12,
    budget_neutral: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Block-randomized on/off flighting schedule for a national series.

    Historical spend is usually smooth and co-moves with demand — the model
    cannot tell the channel from the season. A budget-neutral high/low pulse
    schedule (randomized block order, amplitude ±a%) manufactures exogenous
    variance: identification the data finally pays for. Block length should be
    >= the channel's adstock memory so carryover doesn't smear the contrast.
    """
    if not 0 < amplitude_pct <= 100:
        raise ValueError("amplitude_pct must be in (0, 100]")
    frame = load_design_frame(dataset_path, kpi, channel)
    spend = frame["spend_national"].to_numpy(float)
    kpi_series = frame["kpi_national"].to_numpy(float)

    rng = np.random.default_rng(seed)
    n_blocks = max(2, math.ceil(duration / block_weeks))
    if n_blocks % 2:
        n_blocks += 1  # balanced high/low when budget-neutral
    a = amplitude_pct / 100.0

    # Randomize block order in balanced (high, low) pairs: every consecutive
    # pair contains one high and one low block, coin-flipped — local balance
    # keeps the contrast orthogonal to slow trend while staying unpredictable.
    block_mults: list[float] = []
    for _ in range(n_blocks // 2):
        hi_first = rng.random() < 0.5
        block_mults.extend([1 + a, 1 - a] if hi_first else [1 - a, 1 + a])
    schedule = [
        {"week_offset": w, "multiplier": float(block_mults[w // block_weeks])}
        for w in range(n_blocks * block_weeks)
    ][: max(duration, block_weeks * 2)]
    mults = np.array([s["multiplier"] for s in schedule])
    if budget_neutral:
        mults = mults / mults.mean()  # exact neutrality after truncation
        for s, m in zip(schedule, mults):
            s["multiplier"] = float(round(m, 4))

    # Identification gain. Historical spend variance is mostly ENDOGENOUS
    # (planned around demand/seasonality — the model can't separate it from
    # the baseline), so raw CV comparisons mislead. What the schedule buys is
    # exogenous, randomized variance: report the induced CV and the share of
    # test-window spend variance that is experimenter-controlled.
    mean_spend = float(spend.mean())
    hist_cv = float(spend.std() / max(mean_spend, 1e-9))
    window_cv = float(mults.std())  # multiplier sd ≈ induced CV around the mean
    exogenous_share = float(window_cv**2 / max(window_cv**2 + hist_cv**2, 1e-12))

    # On/off contrast power: ROAS estimated as ΔKPI/Δspend between high and
    # low weeks; KPI noise from the detrended national series.
    sigma_y = _diff_noise_sd(kpi_series)
    t_hi = int((mults > 1).sum())
    t_lo = int((mults < 1).sum())
    spread = float(mults[mults > 1].mean() - mults[mults < 1].mean())
    weekly_spend_delta = mean_spend * spread
    se_roas = (
        sigma_y
        * math.sqrt(1.0 / max(t_hi, 1) + 1.0 / max(t_lo, 1))
        / max(weekly_spend_delta, 1e-12)
    )

    analysis_plan = (
        f"Estimator: on/off contrast (high-block vs low-block weeks) or "
        f"interrupted time-series regression with the schedule as the "
        f"instrument; intention-to-treat on the PLANNED schedule. "
        f"Pre-register: {len(schedule)}-week window, ±{amplitude_pct:.0f}% "
        f"amplitude in {block_weeks}-week blocks, budget-neutral. Leave a "
        f"washout of one adstock window after each block boundary when "
        f"estimating, or model the carryover explicitly. The same weeks feed "
        f"the next MMM fit, where the manufactured variance tightens the "
        f"channel's posterior directly."
    )

    return {
        "design_key": "national_flighting",
        "design_type": "randomized flighting (budget-neutral spend pulses)",
        "channel": channel,
        "kpi": kpi,
        "amplitude_pct": float(amplitude_pct),
        "block_weeks": int(block_weeks),
        "duration": int(len(schedule)),
        "budget_neutral": bool(budget_neutral),
        "schedule": schedule,
        "identification": {
            "historical_spend_cv": hist_cv,
            "scheduled_window_cv": window_cv,
            # share of test-window spend variance that is randomized (clean)
            # rather than endogenous (confounded with demand)
            "exogenous_share": exogenous_share,
        },
        "weekly_spend_delta": float(weekly_spend_delta),
        "se_roas": float(se_roas),
        "mde_roas": float(MDE_FACTOR * se_roas),
        "analysis_plan": analysis_plan,
        "seed": int(seed),
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────


def design_options(dataset_path: str, kpi: str, channel: str) -> dict[str, Any]:
    """What designs the data supports: geo designs need >= 4 geographies."""
    frame = load_design_frame(dataset_path, kpi, channel)
    n_geos = len(frame["geos"])
    has_geo = n_geos >= 4
    return {
        "n_geos": n_geos,
        "geos": frame["geos"],
        "n_weeks": len(frame["periods"]),
        "designs": (
            ["geo_lift", "matched_market_did", "national_flighting"]
            if has_geo
            else ["national_flighting"]
        ),
        "recommended": "geo_lift" if has_geo else "national_flighting",
    }


def design_experiment(
    dataset_path: str,
    kpi: str,
    channel: str,
    *,
    design_key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch to the right design family (auto-recommended when omitted)."""
    if design_key is None:
        design_key = design_options(dataset_path, kpi, channel)["recommended"]
    if design_key == "geo_lift":
        return geo_lift_design(dataset_path, kpi, channel, randomize=True, **kwargs)
    if design_key == "matched_market_did":
        return geo_lift_design(dataset_path, kpi, channel, randomize=False, **kwargs)
    if design_key == "national_flighting":
        return flighting_design(dataset_path, kpi, channel, **kwargs)
    raise ValueError(
        f"Unknown design '{design_key}'. Valid: geo_lift, matched_market_did, "
        "national_flighting"
    )
