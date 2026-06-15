"""A/A and A/B simulation on historical data — to compare experiment
methodologies on their REAL false-positive rate, power, MDE, and cost.

The pure-data designer reports an analytic MDE from a first-difference noise
model. But real KPI series are autocorrelated, so that analytic standard error
understates the spread of the estimator and the nominal false-positive rate
inflates — sometimes several-fold. The only honest way to know how a
methodology behaves on THIS data is to run it on the data:

- **A/A** — slide the design's estimator over historical windows where no
  treatment was applied. The estimator should return ~0; the spread of its
  estimates is the true null distribution. We report the empirical
  false-positive rate at the analytic decision rule (the inflation diagnostic)
  AND a design-calibrated critical value — the empirical ``1-alpha`` quantile of
  ``|estimate|`` — that restores the nominal size. An estimator whose A/A FPR
  far exceeds alpha is INVALID for this data regardless of its nominal power.

- **A/B** — inject a KNOWN lift (the model's predicted incremental KPI when a
  model is available, otherwise a fixed magnitude) onto the treatment arm of
  real historical windows and re-run the estimator. Sweeping a grid of effect
  sizes gives an EMPIRICAL power curve and MDE that respect the real noise
  structure. Power is scored at the calibrated critical value, so it is power at
  the correct size — not a number inflated by the same SE bug.

``methodology_leaderboard`` runs A/A + A/B for several estimators on the SAME
fixed assignment (so the comparison isolates the estimator, not the
randomization draw) and ranks them: valid (FPR within tolerance), powered
(empirical MDE below the expected effect), and cheap (opportunity cost).

numpy/pandas only — kernel-safe. Reuses ``planning.design`` for the panel,
matching, and SE helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .design import (
    _diff_noise_sd,
    _did_se_total_kpi,
    load_design_frame,
    matched_pairs,
    residualize_geo_panel,
)

_Z_975 = 1.959963984540054
_Z_80 = 0.8416212335729143
_EPS = 1e-9


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    pre_slice: slice
    test_slice: slice
    t_pre: int
    t_test: int


@dataclass(frozen=True)
class Assignment:
    """A FROZEN treatment design, reused across every estimator so the
    methodology comparison isolates the estimator (SIM-6)."""

    kind: str  # 'geo' | 'national'
    treatment_geos: tuple[str, ...] = ()
    control_geos: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    schedule_mult: tuple[float, ...] = ()
    seed: int = 42


@dataclass(frozen=True)
class EstimatorResult:
    estimate: float  # total incremental KPI over the test window
    se: float | None  # analytic SE (KPI units); None → null is bootstrap/permutation
    spend_delta: float
    n_eff: int | None


@dataclass(frozen=True)
class AAResult:
    estimator: str
    n_windows: int
    n_eff_windows: int
    null_mean: float
    null_sd: float
    crit_value: float  # design-calibrated (1-alpha) quantile of |estimate|
    fpr_at_nominal: float  # |est|/se > z — the inflation diagnostic
    fpr_at_crit: float  # ≈ alpha by construction (sanity)
    fpr_ci: tuple[float, float]
    fpr_inflated: bool
    fpr_tolerance: float
    null_method: str
    status: str  # 'ok' | 'insufficient_windows'


@dataclass(frozen=True)
class ABResult:
    estimator: str
    per_effect: tuple[dict, ...]
    empirical_mde: float | None  # KPI-total units
    empirical_mde_roas: float | None
    mde_method: str
    mde_ci: tuple[float, float] | None
    expected_effect: float  # KPI-total units (scale=1)
    power_at_expected: float
    null_sd: float
    status: str


@dataclass(frozen=True)
class SimPanel:
    kpi_wide: pd.DataFrame | None
    spend_wide: pd.DataFrame | None
    kpi_national: pd.Series
    spend_national: pd.Series
    residuals: pd.DataFrame | None
    periods: list
    geos: list


def build_sim_panel(dataset_path: str, kpi: str, channel: str) -> SimPanel:
    frame = load_design_frame(dataset_path, kpi, channel)
    kpi_wide = frame["kpi_wide"]
    spend_wide = frame["spend_wide"]
    residuals = None
    if kpi_wide is not None and kpi_wide.shape[1] >= 2:
        residuals = residualize_geo_panel(kpi_wide, spend_wide)["residuals"]
    return SimPanel(
        kpi_wide=kpi_wide,
        spend_wide=spend_wide,
        kpi_national=frame["kpi_national"],
        spend_national=frame["spend_national"],
        residuals=residuals,
        periods=frame["periods"],
        geos=frame["geos"],
    )


# ── Estimators (pure; read only assignment-named rows) ────────────────────────


def _pair_diff_series(panel: SimPanel, assignment: Assignment) -> np.ndarray:
    """Mean over pairs of (treatment - control) KPI per week — the pooled DiD
    response (matches design.geo_lift_design's diff construction)."""
    kpi = panel.kpi_wide
    diffs = [
        kpi[t].to_numpy(float) - kpi[c].to_numpy(float) for t, c in assignment.pairs
    ]
    return np.mean(diffs, axis=0)


def pooled_did_estimator(
    panel: SimPanel, assignment: Assignment, window: Window
) -> EstimatorResult:
    diff = _pair_diff_series(panel, assignment)
    pre, test = diff[window.pre_slice], diff[window.test_slice]
    est = window.t_test * (test.mean() - pre.mean())
    sigma_d = _diff_noise_sd(pre)  # noise from the pre-period only
    se = _did_se_total_kpi(sigma_d, window.t_test, window.t_pre)
    return EstimatorResult(float(est), float(se), 0.0, window.t_test)


def per_pair_did_estimator(
    panel: SimPanel, assignment: Assignment, window: Window
) -> EstimatorResult:
    kpi = panel.kpi_wide
    per_pair = []
    for t, c in assignment.pairs:
        d = kpi[t].to_numpy(float) - kpi[c].to_numpy(float)
        per_pair.append(
            window.t_test * (d[window.test_slice].mean() - d[window.pre_slice].mean())
        )
    per_pair = np.asarray(per_pair, dtype=float)
    est = float(per_pair.mean())
    # Cluster-robust SE: between-pair spread of the per-pair estimate.
    se = (
        float(per_pair.std(ddof=1) / math.sqrt(per_pair.size))
        if per_pair.size > 1
        else None
    )
    return EstimatorResult(est, se, 0.0, per_pair.size)


def regadj_geo_estimator(
    panel: SimPanel, assignment: Assignment, window: Window, *, ridge: float = 1.0
) -> EstimatorResult:
    """Synthetic-control style: fit control weights on the pre-period, score the
    test-window gap. Robust to between-geo scale differences the pooled DiD
    ignores."""
    kpi = panel.kpi_wide
    y = kpi[list(assignment.treatment_geos)].sum(axis=1).to_numpy(float)
    x = kpi[list(assignment.control_geos)].to_numpy(float)  # weeks x n_control
    x_pre, y_pre = x[window.pre_slice], y[window.pre_slice]
    x_test, y_test = x[window.test_slice], y[window.test_slice]
    # ridge solve w = (X'X + lambda I)^-1 X'y with an intercept column
    xp = np.column_stack([np.ones(len(x_pre)), x_pre])
    reg = ridge * np.eye(xp.shape[1])
    reg[0, 0] = 0.0  # don't penalize the intercept
    try:
        w = np.linalg.solve(xp.T @ xp + reg, xp.T @ y_pre)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(xp, y_pre, rcond=None)
    pred_test = np.column_stack([np.ones(len(x_test)), x_test]) @ w
    est = float(np.sum(y_test - pred_test))
    resid = y_pre - xp @ w
    sigma = float(np.std(resid)) if resid.size > 1 else float("nan")
    se = sigma * math.sqrt(window.t_test)
    return EstimatorResult(est, float(se), 0.0, window.t_test)


def national_onoff_estimator(
    panel: SimPanel, assignment: Assignment, window: Window
) -> EstimatorResult:
    """High-block vs low-block KPI contrast over the test window, using the
    design's on/off labels. On un-injected history the contrast is null."""
    kpi = panel.kpi_national.to_numpy(float)
    mult = np.asarray(assignment.schedule_mult, dtype=float)
    test = kpi[window.test_slice]
    t = min(len(test), len(mult))
    if t < 2:
        return EstimatorResult(0.0, None, 0.0, 0)
    test, m = test[:t], mult[:t]
    hi, lo = test[m > 1], test[m < 1]
    if hi.size == 0 or lo.size == 0:
        return EstimatorResult(0.0, None, 0.0, 0)
    contrast = hi.mean() - lo.mean()
    # The total incremental KPI is the per-on-week step times the number of "on"
    # weeks (the lift is injected only on the on-weeks), NOT t_test — scaling by
    # t_test would double-count it ~2x on a balanced schedule. est and se scale
    # by the same factor, so the A/A inflation diagnostic is unchanged.
    t_hi = int(hi.size)
    est = t_hi * contrast
    sigma_y = _diff_noise_sd(kpi[window.pre_slice])
    se = (
        t_hi * sigma_y * math.sqrt(1.0 / hi.size + 1.0 / lo.size)
        if math.isfinite(sigma_y)
        else None
    )
    return EstimatorResult(float(est), None if se is None else float(se), 0.0, t)


# ── Assignment construction ───────────────────────────────────────────────────


def build_geo_assignment(
    panel: SimPanel, *, n_pairs: int | None = None, seed: int = 42
) -> Assignment:
    pairs_raw = matched_pairs(panel.kpi_wide, n_pairs, spend_wide=panel.spend_wide)
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str]] = []
    for p in pairs_raw:
        a, b = p["geo_a"], p["geo_b"]
        t, c = (a, b) if rng.random() < 0.5 else (b, a)
        pairs.append((t, c))
    return Assignment(
        kind="geo",
        treatment_geos=tuple(t for t, _ in pairs),
        control_geos=tuple(c for _, c in pairs),
        pairs=tuple(pairs),
        seed=seed,
    )


def build_national_assignment(
    *, duration: int, block_weeks: int = 2, amplitude_pct: float = 50.0, seed: int = 42
) -> Assignment:
    rng = np.random.default_rng(seed)
    a = amplitude_pct / 100.0
    n_blocks = max(2, math.ceil(duration / block_weeks))
    if n_blocks % 2:
        n_blocks += 1
    mults: list[float] = []
    for _ in range(n_blocks // 2):
        hi_first = rng.random() < 0.5
        mults.extend([1 + a, 1 - a] if hi_first else [1 - a, 1 + a])
    sched = [mults[w // block_weeks] for w in range(n_blocks * block_weeks)][:duration]
    return Assignment(kind="national", schedule_mult=tuple(sched), seed=seed)


# ── Windows ───────────────────────────────────────────────────────────────────


def _windows(
    n_weeks: int, duration: int, *, min_pre: int | None = None
) -> list[Window]:
    min_pre = min_pre if min_pre is not None else max(duration, 8)
    out: list[Window] = []
    for start in range(min_pre, n_weeks - duration + 1):
        out.append(
            Window(
                pre_slice=slice(0, start),
                test_slice=slice(start, start + duration),
                t_pre=start,
                t_test=duration,
            )
        )
    return out


def _subsample(items: list, k: int, seed: int) -> list:
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx)]


# ── Statistics helpers ────────────────────────────────────────────────────────


def _wilson_ci(k: float, n: float, z: float = _Z_975) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _fpr_tolerance(n_eff: float, alpha: float = 0.05) -> float:
    """Adaptive bar: nominal alpha plus the Monte-Carlo slack of n_eff windows,
    floored at 0.075 (SIM-2)."""
    if n_eff <= 0:
        return 0.075
    mc = 1.645 * math.sqrt(alpha * (1 - alpha) / n_eff)
    return max(0.075, alpha + mc)


# ── A/A ───────────────────────────────────────────────────────────────────────


def run_aa_simulation(
    panel: SimPanel,
    estimator: Callable[[SimPanel, Assignment, Window], EstimatorResult],
    assignment: Assignment,
    *,
    duration: int,
    alpha: float = 0.05,
    max_windows: int = 200,
    seed: int = 42,
    name: str | None = None,
) -> AAResult:
    n_weeks = (
        len(panel.kpi_national)
        if assignment.kind == "national"
        else len(panel.kpi_wide)
    )
    wins = _subsample(_windows(n_weeks, duration), max_windows, seed)
    est_name = name or getattr(estimator, "__name__", "estimator")
    if len(wins) < 4:
        return AAResult(
            est_name,
            len(wins),
            0,
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            alpha,
            (0.0, 1.0),
            False,
            _fpr_tolerance(0, alpha),
            "insufficient",
            "insufficient_windows",
        )

    estimates, reject_nominal = [], []
    for w in wins:
        r = estimator(panel, assignment, w)
        if not math.isfinite(r.estimate):
            continue
        estimates.append(r.estimate)
        if r.se and r.se > _EPS:
            reject_nominal.append(abs(r.estimate) / r.se > _Z_975)
    est = np.asarray(estimates, dtype=float)
    if est.size < 4:
        return AAResult(
            est_name,
            est.size,
            0,
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            alpha,
            (0.0, 1.0),
            False,
            _fpr_tolerance(0, alpha),
            "insufficient",
            "insufficient_windows",
        )

    n_eff = max(1, int(round((n_weeks - duration) / max(duration, 1))))
    crit = float(np.quantile(np.abs(est), 1 - alpha))
    null_method = "analytic_se" if reject_nominal else "calibrated_quantile"
    fpr_nominal = float(np.mean(reject_nominal)) if reject_nominal else float("nan")
    fpr_crit = float(np.mean(np.abs(est) > crit))
    tol = _fpr_tolerance(n_eff, alpha)
    fpr_for_ci = fpr_nominal if reject_nominal else fpr_crit
    ci = _wilson_ci(fpr_for_ci * n_eff, n_eff)
    inflated = bool(reject_nominal and np.isfinite(fpr_nominal) and fpr_nominal > tol)
    # Graded confidence: too few effective windows to estimate anything (<8),
    # limited (8-30, wide CI — the adaptive tolerance compensates), or ok.
    if n_eff < 8:
        status = "insufficient_windows"
    elif n_eff < 30:
        status = "limited"
    else:
        status = "ok"

    return AAResult(
        estimator=est_name,
        n_windows=int(est.size),
        n_eff_windows=n_eff,
        null_mean=float(est.mean()),
        null_sd=float(est.std()),
        crit_value=crit,
        fpr_at_nominal=fpr_nominal,
        fpr_at_crit=fpr_crit,
        fpr_ci=ci,
        fpr_inflated=inflated,
        fpr_tolerance=tol,
        null_method=null_method,
        status=status,
    )


# ── Lift injection ────────────────────────────────────────────────────────────


@dataclass
class LiftInjector:
    """Adds a known lift to the treatment arm of a window. ``expected_total`` is
    the lift magnitude at scale=1 (model-predicted when anchored). ``__call__``
    returns ``(true_total, injected_kpi_wide)`` for the geo path or
    ``(true_total, injected_kpi_national)`` for national."""

    kind: str  # 'model_anchored' | 'spend_share' | 'uniform'
    expected_total: float
    per_draw_total: np.ndarray | None = None
    spend_share: dict[str, float] | None = None
    per_draw_total_list: list = field(default_factory=list)

    def inject_geo(
        self, panel: SimPanel, assignment: Assignment, window: Window, scale: float
    ) -> tuple[float, pd.DataFrame]:
        kpi = panel.kpi_wide.copy()
        true_total = scale * self.expected_total
        treat = list(assignment.treatment_geos)
        rows = kpi.index[window.test_slice]
        if self.kind == "spend_share" and panel.spend_wide is not None:
            shares = np.array(
                [max(self.spend_share.get(g, 0.0), 0.0) for g in treat], dtype=float
            )
            shares = shares / shares.sum() if shares.sum() > _EPS else None
        else:
            shares = None
        for gi, g in enumerate(treat):
            share = shares[gi] if shares is not None else 1.0 / len(treat)
            # additive, spread evenly across the test weeks of this geo cell
            kpi.loc[rows, g] = kpi.loc[rows, g] + (true_total * share) / window.t_test
        return true_total, kpi

    def inject_national(
        self, panel: SimPanel, assignment: Assignment, window: Window, scale: float
    ) -> tuple[float, pd.Series]:
        kpi = panel.kpi_national.copy()
        true_total = scale * self.expected_total
        mult = np.asarray(assignment.schedule_mult, dtype=float)
        rows = kpi.index[window.test_slice]
        t = min(len(rows), len(mult))
        weights = np.maximum(mult[:t] - 1.0, 0.0)  # lift only on the "on" weeks
        weights = weights / weights.sum() if weights.sum() > _EPS else np.ones(t) / t
        add = true_total * weights
        vals = kpi.loc[rows].to_numpy(float)
        vals[:t] = vals[:t] + add
        kpi.loc[rows] = vals
        return true_total, kpi


def fixed_lift_injector(
    panel: SimPanel,
    assignment: Assignment,
    *,
    duration: int,
    pct_of_baseline: float | None = None,
    kpi_units: float | None = None,
) -> LiftInjector:
    """Model-free injector: lift = ``kpi_units`` or ``pct_of_baseline`` of the
    treatment arm's mean window KPI. Distributed by spend share (clipped)."""
    if assignment.kind == "national":
        base = float(panel.kpi_national.tail(duration).mean()) * duration
    else:
        treat = list(assignment.treatment_geos)
        base = float(panel.kpi_wide[treat].tail(duration).sum(axis=1).mean()) * duration
    if kpi_units is not None:
        total = float(kpi_units)
    else:
        total = float((pct_of_baseline or 0.05) * base)
    shares = None
    if assignment.kind == "geo" and panel.spend_wide is not None:
        shares = {
            g: float(panel.spend_wide[g].sum()) if g in panel.spend_wide else 0.0
            for g in assignment.treatment_geos
        }
    return LiftInjector(
        kind="spend_share" if shares else "uniform",
        expected_total=total,
        spend_share=shares,
    )


# ── A/B ───────────────────────────────────────────────────────────────────────


def _probit_mde(
    effects: np.ndarray, powers: np.ndarray, *, target: float = 0.8
) -> tuple[float | None, str]:
    """Smallest effect with ``target`` power via a probit fit on the
    power-vs-effect curve (isotonic-cleaned). Linear-interp fallback < 3 points."""
    mask = (powers > 1e-3) & (powers < 1 - 1e-3) & np.isfinite(effects)
    e, p = effects[mask], powers[mask]
    if e.size >= 3:
        order = np.argsort(e)
        e, p = e[order], p[order]
        z = np.array([_norm_ppf(v) for v in p])
        A = np.column_stack([np.ones_like(e), e])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        a, b = coef
        if abs(b) > _EPS:
            mde = (_norm_ppf(target) - a) / b
            if mde > 0:
                return float(mde), "probit_fit"
    # fallback: first effect whose power >= target by linear interp on full grid
    full_mask = np.isfinite(effects)
    ef, pf = effects[full_mask], powers[full_mask]
    if ef.size == 0:
        return None, "interp_fallback"
    order = np.argsort(ef)
    ef, pf = ef[order], pf[order]
    # Already powered at the smallest tested effect → the true MDE is at or below
    # the grid floor; report ef[0] as a conservative MDE rather than None (which
    # would mislabel a strongly-powered estimator as not powered).
    if pf[0] >= target:
        return float(ef[0]), "at_floor"
    for i in range(1, len(ef)):
        if pf[i - 1] < target <= pf[i]:
            t = (target - pf[i - 1]) / max(pf[i] - pf[i - 1], _EPS)
            return float(ef[i - 1] + t * (ef[i] - ef[i - 1])), "interp_fallback"
    # never reaches target on the grid → genuinely underpowered
    return None, "underpowered_at_grid"


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def run_ab_simulation(
    panel: SimPanel,
    estimator: Callable[[SimPanel, Assignment, Window], EstimatorResult],
    assignment: Assignment,
    injector: LiftInjector,
    *,
    duration: int,
    aa_result: AAResult,
    effect_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    spend_delta_window: float = 0.0,
    estimand_scale: float = 1.0,
    max_windows: int = 120,
    seed: int = 42,
    name: str | None = None,
) -> ABResult:
    est_name = name or getattr(estimator, "__name__", "estimator")
    n_weeks = (
        len(panel.kpi_national)
        if assignment.kind == "national"
        else len(panel.kpi_wide)
    )
    wins = _subsample(_windows(n_weeks, duration), max_windows, seed)
    crit = aa_result.crit_value
    if not math.isfinite(crit) or len(wins) < 4:
        return ABResult(
            est_name,
            (),
            None,
            None,
            "insufficient",
            None,
            injector.expected_total,
            float("nan"),
            aa_result.null_sd,
            "insufficient_windows",
        )

    per_effect = []
    for scale in effect_grid:
        n_sig, n_signed, biases, covered = 0, 0, [], 0
        true_total = scale * injector.expected_total
        # The injector adds the FULL total across treated cells, but each
        # estimator's natural estimand is a fraction/multiple of that (a pooled
        # DiD averages over pairs → total/n_pairs). bias/coverage/rmse must be
        # scored against THIS estimator's own estimand, else an unbiased
        # estimator looks grossly biased (the power axis is scale-free via the
        # A/A crit, so it is left alone).
        cmp_total = estimand_scale * true_total
        n = 0
        for w in wins:
            inj_panel = _apply_injection(panel, assignment, injector, w, scale)
            r = estimator(inj_panel, assignment, w)
            if not math.isfinite(r.estimate):
                continue
            n += 1
            sig = abs(r.estimate) > crit
            if sig:
                n_sig += 1
                if np.sign(r.estimate) == np.sign(true_total) or true_total == 0:
                    n_signed += 1
            biases.append(r.estimate - cmp_total)
            if r.se and r.se > _EPS:
                if abs(r.estimate - cmp_total) <= _Z_975 * r.se:
                    covered += 1
        if n == 0:
            continue
        power = n_signed / n
        per_effect.append(
            {
                "effect": float(true_total),
                "scale": float(scale),
                "power": float(power),
                "power_ci": list(_wilson_ci(n_signed, n)),
                "bias": float(np.mean(biases)) if biases else None,
                "rmse": float(np.sqrt(np.mean(np.square(biases)))) if biases else None,
                "coverage": float(covered / n) if n else None,
                "n": int(n),
            }
        )

    if not per_effect:
        return ABResult(
            est_name,
            (),
            None,
            None,
            "insufficient",
            None,
            injector.expected_total,
            float("nan"),
            aa_result.null_sd,
            "insufficient_windows",
        )
    effects = np.array([e["effect"] for e in per_effect])
    powers = np.array([e["power"] for e in per_effect])
    mde, mde_method = _probit_mde(np.abs(effects), powers)
    mde_roas = (
        mde / abs(spend_delta_window)
        if (mde is not None and abs(spend_delta_window) > _EPS)
        else None
    )
    # power at the expected effect (scale closest to 1.0)
    i1 = int(np.argmin([abs(e["scale"] - 1.0) for e in per_effect]))
    return ABResult(
        estimator=est_name,
        per_effect=tuple(per_effect),
        empirical_mde=mde,
        empirical_mde_roas=mde_roas,
        mde_method=mde_method,
        mde_ci=None,
        expected_effect=float(injector.expected_total),
        power_at_expected=float(per_effect[i1]["power"]),
        null_sd=aa_result.null_sd,
        status="ok",
    )


def _apply_injection(panel, assignment, injector, window, scale):
    if assignment.kind == "national":
        _t, kpi = injector.inject_national(panel, assignment, window, scale)
        return SimPanel(
            panel.kpi_wide,
            panel.spend_wide,
            kpi,
            panel.spend_national,
            panel.residuals,
            panel.periods,
            panel.geos,
        )
    _t, kpi_wide = injector.inject_geo(panel, assignment, window, scale)
    return SimPanel(
        kpi_wide,
        panel.spend_wide,
        panel.kpi_national,
        panel.spend_national,
        None,
        panel.periods,
        panel.geos,
    )


# ── Leaderboard ───────────────────────────────────────────────────────────────

_GEO_ESTIMATORS = {
    "pooled_did": pooled_did_estimator,
    "per_pair_did": per_pair_did_estimator,
    "regadj_geo": regadj_geo_estimator,
}
_NATIONAL_ESTIMATORS = {"national_onoff": national_onoff_estimator}


def _finite(x: float | None) -> float | None:
    """None for None/NaN/Inf, else the float — so leaderboard rows never carry a
    native non-finite that a strict JSON encoder (JSONResponse) would reject."""
    return float(x) if (x is not None and math.isfinite(x)) else None


def _estimand_scale(key: str, assignment: Assignment) -> float:
    """Each estimator recovers a known fraction/multiple of the injected total:
    the pooled / per-pair DiD average over pairs (→ total / n_pairs); the
    synthetic-control and (fixed) national on/off recover the full total."""
    if key in ("pooled_did", "per_pair_did"):
        return 1.0 / max(len(assignment.pairs), 1)
    return 1.0


def methodology_leaderboard(
    dataset_path: str,
    kpi: str,
    channel: str,
    *,
    mmm: Any = None,
    design: dict | None = None,
    duration: int = 8,
    alpha: float = 0.05,
    target_mde_roas: float | None = None,
    spend_delta_window: float = 0.0,
    expected_effect_total: float | None = None,
    fixed_pct_of_baseline: float = 0.1,
    max_aa_windows: int = 200,
    max_ab_windows: int = 120,
    seed: int = 42,
) -> dict[str, Any]:
    """Run A/A + A/B for every estimator the data supports on ONE fixed
    assignment, and rank them by validity → power → cost."""
    panel = build_sim_panel(dataset_path, kpi, channel)
    geo = panel.kpi_wide is not None and panel.kpi_wide.shape[1] >= 4

    if geo:
        assignment = build_geo_assignment(panel, seed=seed)
        estimators = _GEO_ESTIMATORS
    else:
        assignment = build_national_assignment(duration=duration, seed=seed)
        estimators = _NATIONAL_ESTIMATORS

    # Injector: model-anchored magnitude when a model is available, else fixed.
    if expected_effect_total is not None and abs(expected_effect_total) > _EPS:
        injector = LiftInjector(
            kind="model_anchored", expected_total=abs(float(expected_effect_total))
        )
        if assignment.kind == "geo" and panel.spend_wide is not None:
            injector.spend_share = {
                g: float(panel.spend_wide[g].sum()) if g in panel.spend_wide else 0.0
                for g in assignment.treatment_geos
            }
            injector.kind = "spend_share"
        injection_basis = "model_anchored"
    else:
        injector = fixed_lift_injector(
            panel, assignment, duration=duration, pct_of_baseline=fixed_pct_of_baseline
        )
        injection_basis = injector.kind

    rows = []
    for key, est in estimators.items():
        aa = run_aa_simulation(
            panel,
            est,
            assignment,
            duration=duration,
            alpha=alpha,
            max_windows=max_aa_windows,
            seed=seed,
            name=key,
        )
        ab = run_ab_simulation(
            panel,
            est,
            assignment,
            injector,
            duration=duration,
            aa_result=aa,
            spend_delta_window=spend_delta_window,
            estimand_scale=_estimand_scale(key, assignment),
            max_windows=max_ab_windows,
            seed=seed,
            name=key,
        )
        usable = aa.status != "insufficient_windows"
        valid = (
            (
                usable
                and np.isfinite(aa.fpr_at_nominal)
                and aa.fpr_at_nominal <= aa.fpr_tolerance
            )
            if aa.null_method == "analytic_se"
            else usable
        )
        powered = (
            ab.empirical_mde_roas is not None
            and target_mde_roas is not None
            and ab.empirical_mde_roas <= target_mde_roas
        )
        rows.append(
            {
                "key": key,
                "label": key.replace("_", " "),
                "valid": bool(valid),
                "fpr": _finite(aa.fpr_at_nominal),
                "fpr_at_crit": _finite(aa.fpr_at_crit),
                "fpr_ci": [_finite(aa.fpr_ci[0]), _finite(aa.fpr_ci[1])],
                "fpr_inflated": aa.fpr_inflated,
                "fpr_tolerance": _finite(aa.fpr_tolerance),
                "n_eff_windows": aa.n_eff_windows,
                "null_method": aa.null_method,
                "null_sd": _finite(aa.null_sd),
                "crit_value": _finite(aa.crit_value),
                "empirical_mde": _finite(ab.empirical_mde),
                "empirical_mde_roas": _finite(ab.empirical_mde_roas),
                "mde_method": ab.mde_method,
                "power_at_expected_effect": _finite(ab.power_at_expected),
                "powered": bool(powered),
                "aa_status": aa.status,
                "ab_status": ab.status,
                "power_curve": [
                    {"effect": e["effect"], "power": e["power"], "scale": e["scale"]}
                    for e in ab.per_effect
                ],
            }
        )

    # Rank: valid first, then powered, then lowest empirical MDE (ROAS).
    def _rank(r: dict) -> tuple:
        mde = r["empirical_mde_roas"]
        return (
            0 if r["valid"] else 1,
            0 if r["powered"] else 1,
            mde if mde is not None else float("inf"),
        )

    rows.sort(key=_rank)
    chosen = rows[0]["key"] if rows and rows[0]["valid"] else None

    return {
        "alpha": alpha,
        "duration": int(duration),
        "kind": assignment.kind,
        "injection_basis": injection_basis,
        "expected_effect": float(injector.expected_total),
        "spend_delta_window": float(spend_delta_window),
        "methodologies": rows,
        "chosen_key": chosen,
        "caveats": [
            "Sliding windows overlap, so the reported rates are indicative, not "
            "independent Bernoulli trials — see n_eff_windows.",
            "False-positive rate is measured against the analytic decision rule; "
            "the calibrated critical value restores nominal size for the power test.",
        ],
    }
