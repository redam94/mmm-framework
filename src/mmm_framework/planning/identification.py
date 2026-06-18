"""Structural-parameter identification design for multi-level flighting.

The reduced-form flighting power (``planning.design.flighting_estimand_ses``)
answers *can the test measure the contribution / average-ROAS / marginal-ROAS*.
This module answers a different, deeper question the client asked for: *will the
next MMM refit, fed the manufactured exogenous spend variance, actually move the
channel's STRUCTURAL parameters* — the response-curve shape ``psi`` (saturation),
the carryover ``alpha`` (adstock), and the coefficient ``beta`` — off their
priors? Per ``transforms.adstock`` §3.6 those three are entangled on an
equifinality ridge and only weakly identified from smooth observational spend; a
schedule that *spans spend levels* (curvature → ``psi``) and *pulses sharply*
(temporal contrast → ``alpha``) is what breaks the ridge.

Method (a LOCAL Laplace design, **kernel-safe** — numpy + ``transforms`` only,
no model/pymc import; the fitted-model anchor is extracted by the caller and
passed in as plain arrays):

* Reconstruct the model's exact forward op on the NORMALIZED basis
  ``x_norm = mult * op_spend / raw_max`` → normalized-FIR adstock → logistic
  saturation ``1 - exp(-clip(lam*a, 0, 20))`` → ``beta * s * y_std`` (original
  KPI scale). This byte-mirrors ``BayesianMMM.sample_channel_contributions`` for
  the v1-gated geometric + logistic parametric path.
* Build the per-week Jacobian ``J_t = dc_t/dtheta`` (analytic ``dc/dbeta``;
  central finite differences for ``dc/dalpha`` and ``dc/dpsi`` THROUGH the real
  ``adstock_weights``+``apply_adstock``, never the infinite-recursion analytic
  derivative). Take it over the EXPERIMENT-WINDOW INCREMENTAL rows only — the
  flighting contrast vs business-as-usual (``J_t`` minus the BAU row) — so a flat
  schedule contributes nothing and the historical data is never double-counted.
* PROFILE OUT the nuisance baseline (intercept + linear trend + yearly Fourier)
  by residualizing every Jacobian column against it, so a schedule collinear
  with trend/season can't earn phantom identification.
* Experiment Fisher ``F = (1/sigma^2) J_res^T J_res``; posterior precision
  ``F_post = F + diag(1/prior_var)`` (a DIAGONAL prior-precision ridge from the
  robust draw-based widths — NEVER ``inv(cov)`` of the skewed, banana-shaped
  structural posterior). Per-parameter posterior SD = ``sqrt(diag(inv(F_post)))``.

Honesty rails (all from the locked design review): identification is driven by
the smallest eigenvalue / per-parameter contraction (E-optimality), NEVER the
determinant; a parameter is only "claimed" when the design actually excites it
(``alpha`` only with temporal contrast, ``psi`` only with >=3 in-support levels);
the result is an OPTIMISTIC UPPER BOUND on what a refit achieves and must never
solely drive a Pareto objective; degenerate / near-zero-gradient / over-condition
designs return ``None`` rather than a falsely-tight CI.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..transforms.adstock import adstock_weights, apply_adstock

# z_{0.975} + z_{0.80} = 2.8 — the MDE factor (mirrors design.MDE_FACTOR).
_FACTOR = 2.8
_Z_975 = 1.959963984540054
_EPS = 1e-12
# Logistic saturation clamps the exponent -lam*a to [-20, 0] in-graph, so the
# model's own gradient is exactly 0 where lam*a > 20 (fully saturated).
_SAT_CLAMP = 20.0
# A parameter whose incremental-Jacobian column norm is below this fraction of
# the beta column's norm is not meaningfully excited by the design.
_EXCITE_TOL = 1e-3
# Minimum posterior contraction (1 - post_sd/prior_sd) to call a claimed
# parameter "identified" by the experiment.
_MIN_CONTRACTION = 0.05
# A parameter is only "claimed" (i.e. the design provides real information about
# it, and it may bind the headline) when the experiment moves its posterior at
# least this much. Far below the real-design regime (~0.03-0.05) but far above
# the near-flat/degenerate regime (~1e-5), so a zero-information schedule claims
# nothing and never reports a prior-driven, falsely-confident binding power.
_MIN_CLAIM_CONTRACTION = 1e-3
_MIN_LEVEL_SEP = 0.02  # mirrors design._MIN_LEVEL_SEP


def _robust_sd(x: np.ndarray) -> float:
    """Robust width of a (possibly skewed, banana-ridge) posterior: MAD→sigma
    (``1.4826 * median|x - median(x)|``), falling back to ``np.std`` only when
    the MAD collapses. Used for the prior-precision ridge and the contraction
    denominator so a heavy-tailed structural posterior doesn't inflate the width."""
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sd = 1.4826 * mad
    if not (math.isfinite(sd) and sd > 0):
        sd = float(np.std(x))
    return sd


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard-normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2.0)))


def _power_from_se(se: float, effect_draws: np.ndarray) -> float:
    """Signed two-sided power to resolve a parameter from 0 at standard error
    ``se``: ``mean_d[Phi(theta_d/se - z) + Phi(-theta_d/se - z)]`` over the
    parameter's posterior draws. Identical estimator to the ROAS power
    (``experiment_optimizer._power_from_se``) so the UI columns are on one scale.
    """
    draws = np.asarray(effect_draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0 or not math.isfinite(se) or se < 0:
        return float("nan")
    if se <= _EPS:
        return 1.0
    p = _phi(draws / se - _Z_975) + _phi(-draws / se - _Z_975)
    return float(np.clip(np.mean(p), 0.0, 1.0))


# ── Design-side noise guards (pure numpy; reused by the Tier-1 flighting path) ──


def ar1_design_effect(
    residual: np.ndarray, window: int, *, rho_max: float = 0.95
) -> dict[str, Any]:
    """AR(1) variance design-effect for a length-``window`` average/contrast.

    Autocorrelated residuals inflate the variance of a windowed estimator beyond
    the i.i.d. ``sigma^2/n`` rule by ``deff = 1 + 2*sum_{k=1}^{L-1}(1 - k/L)*rho^k``
    (Newey-West / batch-means form). ``rho`` is the lag-1 autocorrelation of the
    baseline-residualized series, clipped to ``[0, rho_max]``; ``deff`` is capped
    at the window length ``L`` (the fully-correlated bound). Multiply ``sigma`` by
    ``sqrt(deff)`` BEFORE the OLS covariance so an autocorrelated window cannot be
    labelled powered on an i.i.d. SE.
    """
    r = np.asarray(residual, dtype=float)
    r = r[np.isfinite(r)]
    L = int(max(1, window))
    out: dict[str, Any] = {"rho": 0.0, "deff": 1.0, "n": int(r.size)}
    if r.size < 4 or L < 2:
        return out
    r = r - r.mean()
    denom = float(np.sum(r * r))
    if denom <= _EPS:
        return out
    rho = float(np.sum(r[1:] * r[:-1]) / denom)
    rho = float(np.clip(rho, 0.0, rho_max))
    k = np.arange(1, L)
    deff = 1.0 + 2.0 * float(np.sum((1.0 - k / L) * rho**k))
    deff = float(np.clip(deff, 1.0, float(L)))
    out.update(rho=rho, deff=deff)
    return out


def cross_channel_vif(
    target_window_spend: np.ndarray, other_window_spend: np.ndarray | None
) -> dict[str, Any]:
    """Variance-inflation from co-spending channels over the scheduled window.

    Regresses the test channel's SCHEDULED window spend on the other channels'
    actual window spend and returns ``R2`` and the inflation ``1/(1-R2)``. The
    scheduled variance is randomized, but if other channels happen to co-move
    with it over the window the design's effective contrast shrinks. Surface it
    and degrade power when ``R2`` is high; ``other_window_spend`` is
    ``(L, n_other)`` or ``None`` (single-channel → no inflation).
    """
    y = np.asarray(target_window_spend, dtype=float)
    out: dict[str, Any] = {"r2": 0.0, "vif": 1.0}
    if other_window_spend is None:
        return out
    X = np.asarray(other_window_spend, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != y.size or y.size < 3 or not np.all(np.isfinite(X)):
        return out
    y = y - y.mean()
    sst = float(np.sum(y * y))
    if sst <= _EPS:
        return out
    A = np.column_stack([np.ones_like(y), X - X.mean(axis=0)])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = float(np.clip(1.0 - np.sum(resid * resid) / sst, 0.0, 1.0))
    except np.linalg.LinAlgError:
        return out
    vif = float(1.0 / max(1.0 - r2, _EPS)) if r2 < 1.0 else float("inf")
    out.update(r2=r2, vif=vif)
    return out


# ── Forward op (byte-mirror of the gated geometric+logistic graph) ─────────────


def _forward_contribution(
    mults: np.ndarray,
    op_spend: float,
    raw_max: float,
    y_std: float,
    beta: float,
    alpha: float,
    lam: float,
    *,
    l_max: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-week channel contribution (original KPI scale) for a multiplier
    schedule, plus the per-week saturated exponent ``lam*a`` (for the clamp mask).

    Mirrors ``BayesianMMM`` parametric geometric + logistic: normalize raw spend
    by ``raw_max``, convolve with the normalized geometric FIR kernel, apply
    logistic saturation with the in-graph exponent clamp, scale by ``beta*y_std``.
    """
    x_norm = (np.asarray(mults, dtype=float) * float(op_spend)) / max(
        float(raw_max), _EPS
    )
    w = adstock_weights("geometric", int(l_max), alpha=float(alpha), normalize=normalize)
    a = apply_adstock(x_norm, w)
    z = np.clip(float(lam) * a, 0.0, _SAT_CLAMP)
    s = 1.0 - np.exp(-z)
    c = float(beta) * s * float(y_std)
    return c, float(lam) * a


def _central_diff(
    f, value: float, step: float, *, lo: float | None = None, hi: float | None = None
) -> np.ndarray:
    """Central finite difference ``(f(value+h) - f(value-h))/(2h)`` with the
    perturbation clamped into ``(lo, hi)`` and the divisor matching the realized
    (possibly asymmetric, post-clamp) span."""
    hi_v = value + step
    lo_v = value - step
    if hi is not None:
        hi_v = min(hi_v, hi)
    if lo is not None:
        lo_v = max(lo_v, lo)
    span = hi_v - lo_v
    if span <= _EPS:
        return np.zeros_like(f(value))
    return (f(hi_v) - f(lo_v)) / span


def _nuisance_basis(n: int) -> np.ndarray:
    """Deterministic baseline regressors over an ``n``-week window: intercept,
    linear trend, and two yearly (52-week) Fourier harmonics — the low-frequency
    structure the model co-estimates and the schedule must be decorrelated from."""
    t = np.arange(n, dtype=float)
    cols = [np.ones(n), (t - t.mean()) / max(t.std(), _EPS)]
    for h in (1, 2):
        cols.append(np.sin(2.0 * np.pi * h * t / 52.0))
        cols.append(np.cos(2.0 * np.pi * h * t / 52.0))
    return np.column_stack(cols)


def _residualize(J: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
    """Project each Jacobian column orthogonal to the nuisance basis (profile out
    the co-estimated baseline). ``J`` is ``(L, P)``; ``basis`` is ``(L, Q)``."""
    if basis is None or basis.size == 0:
        return J
    try:
        coef, *_ = np.linalg.lstsq(basis, J, rcond=None)
    except np.linalg.LinAlgError:
        return J
    return J - basis @ coef


# ── Public API ─────────────────────────────────────────────────────────────────


def structural_jacobian(
    mults: np.ndarray,
    op_spend: float,
    raw_max: float,
    y_std: float,
    theta: dict[str, float],
    *,
    l_max: int = 8,
    normalize: bool = True,
    prior_sd: dict[str, float] | None = None,
    fd_rel: float = 0.01,
    residualize: bool = True,
) -> dict[str, Any] | None:
    """Residualized, BAU-incremental per-week Jacobian of channel contribution
    w.r.t. ``theta = {beta, alpha, lam}`` for a multiplier schedule.

    Columns: analytic ``dc/dbeta = s*y_std``; central finite differences for
    ``dc/dalpha`` and ``dc/dlam`` through the real normalized-FIR adstock +
    logistic saturation. Each column is taken as the CONTRAST vs the BAU
    (all-ones) schedule and residualized against the nuisance basis. Rows where
    the saturation exponent ``lam*a`` is clamped (>20) are zeroed (the model's
    own gradient is exactly 0 there). Returns ``None`` on non-finite/degenerate
    input.
    """
    mults = np.asarray(mults, dtype=float)
    if mults.size < 2 or not np.all(np.isfinite(mults)):
        return None
    beta = float(theta.get("beta", float("nan")))
    alpha = float(theta.get("alpha", float("nan")))
    lam = float(theta.get("lam", float("nan")))
    if not all(math.isfinite(v) for v in (beta, alpha, lam, op_spend, raw_max, y_std)):
        return None
    if op_spend <= 0 or raw_max <= 0 or y_std <= 0 or beta <= 0 or lam <= 0:
        return None
    if not (0.0 <= alpha < 1.0):
        return None

    names = ["beta", "alpha", "lam"]
    psd = prior_sd or {}

    def contrib(b: float, al: float, lm: float, m: np.ndarray) -> np.ndarray:
        c, _ = _forward_contribution(
            m, op_spend, raw_max, y_std, b, al, lm, l_max=l_max, normalize=normalize
        )
        return c

    bau = np.ones_like(mults)
    # Clamp mask at the scheduled point (saturated rows carry no marginal info).
    _, z_sched = _forward_contribution(
        mults, op_spend, raw_max, y_std, beta, alpha, lam, l_max=l_max, normalize=normalize
    )
    clamp_mask = z_sched >= _SAT_CLAMP - _EPS

    # dc/dbeta is analytic and linear: contribution/beta = s*y_std.
    s_sched = contrib(beta, alpha, lam, mults) / beta
    s_bau = contrib(beta, alpha, lam, bau) / beta
    col_beta = s_sched - s_bau

    # dc/dalpha, dc/dlam via central differences (step = fd_rel * prior width,
    # floored to a small fraction of the value so a tiny prior can't zero it).
    h_alpha = max(fd_rel * float(psd.get("alpha", 0.0)), fd_rel * max(alpha, 1e-3))
    h_lam = max(fd_rel * float(psd.get("lam", 0.0)), fd_rel * max(lam, 1e-3))
    col_alpha = _central_diff(
        lambda al: contrib(beta, al, lam, mults) - contrib(beta, al, lam, bau),
        alpha,
        h_alpha,
        lo=0.0,
        hi=1.0 - 1e-6,
    )
    col_lam = _central_diff(
        lambda lm: contrib(beta, alpha, lm, mults) - contrib(beta, alpha, lm, bau),
        lam,
        h_lam,
        lo=1e-9,
    )

    J = np.column_stack([col_beta, col_alpha, col_lam])
    J[clamp_mask, :] = 0.0  # saturated rows: model gradient is exactly 0
    if not np.all(np.isfinite(J)):
        return None

    basis = _nuisance_basis(mults.size) if residualize else None
    J_res = _residualize(J, basis)
    if not np.all(np.isfinite(J_res)):
        return None
    return {
        "J": J_res,
        "J_raw": J,
        "names": names,
        "theta": {"beta": beta, "alpha": alpha, "lam": lam},
        "clamp_mask": clamp_mask,
        "n_clamped": int(clamp_mask.sum()),
    }


def structural_information(
    J: np.ndarray,
    names: list[str],
    sigma: float,
    prior_sd: dict[str, float],
) -> dict[str, Any] | None:
    """Experiment + posterior Fisher information from a residualized Jacobian.

    ``F = (1/sigma^2) J^T J``; ``F_post = F + diag(1/prior_var)`` (a diagonal
    prior-precision ridge from the robust draw widths — never ``inv(cov)`` of the
    structural posterior). Returns per-parameter posterior SD, the smallest
    eigenvalue + its eigenvector (the equifinality direction the design fails to
    pin), the condition number, and a relative information gain. ``None`` if the
    inputs are degenerate.
    """
    J = np.asarray(J, dtype=float)
    P = len(names)
    if J.ndim != 2 or J.shape[1] != P or not math.isfinite(sigma) or sigma <= 0:
        return None
    pv = np.array([float(prior_sd.get(n, float("nan"))) ** 2 for n in names])
    if not np.all(np.isfinite(pv)) or np.any(pv <= 0):
        return None
    prior_prec = np.diag(1.0 / pv)
    F = (J.T @ J) / (sigma * sigma)
    if not np.all(np.isfinite(F)):
        return None
    F_post = F + prior_prec
    try:
        cov = np.linalg.inv(F_post)
    except np.linalg.LinAlgError:
        return None
    var = np.diag(cov)
    if np.any(var <= 0) or not np.all(np.isfinite(var)):
        return None
    post_sd = {n: float(math.sqrt(v)) for n, v in zip(names, var)}
    # E-optimality: smallest eigenvalue of the EXPERIMENT Fisher = the worst-
    # determined direction (the equifinality ridge), and its eigenvector.
    evals, evecs = np.linalg.eigh(F)
    lam_min = float(evals[0])
    eqf_vec = {n: float(v) for n, v in zip(names, evecs[:, 0])}
    cond = float(evals[-1] / evals[0]) if evals[0] > _EPS else float("inf")
    # Relative info gain = product of per-param precision ratios (posterior vs
    # prior), reported as a diagnostic only (never summed with EVOI nats).
    rel_gain = float(np.sum(0.5 * np.log(pv / var)))  # 0.5*log det(F_post * prior_cov)
    return {
        "post_sd": post_sd,
        "lambda_min": lam_min,
        "equifinality_vec": eqf_vec,
        "condition": cond,
        "rel_gain_nats": rel_gain,
        "F": F,
    }


def structural_identification(
    mults: np.ndarray,
    op_spend: float,
    raw_max: float,
    y_std: float,
    theta_draws: dict[str, np.ndarray],
    *,
    sigma_lo: float,
    sigma_hi: float,
    l_max: int = 8,
    normalize: bool = True,
    in_support: bool = True,
    power_target: float = 0.80,
    fd_rel: float = 0.01,
) -> dict[str, Any] | None:
    """Per-parameter identification power / MDE / contraction for a multi-level
    flighting schedule — the structural (beta, alpha, psi=lam) block.

    OPTIMISTIC UPPER BOUND on what the next refit achieves; never let it solely
    drive a Pareto objective. ``theta_draws`` are the channel's flattened
    posterior draws (``beta``, ``alpha``, ``lam``); ``theta_hat`` is the median.
    ``sigma_lo`` (model sigma*y_std, optimistic floor) and ``sigma_hi``
    (regression residual, pessimistic) bracket the noise; the headline power uses
    the conservative ``max(sigma_lo, sigma_hi)``. A parameter is "claimed" only
    when the design is structurally eligible AND the experiment actually
    contracts it (an absolute floor): ``beta``/``alpha`` need real contrast,
    ``lam`` needs >=3 in-support distinct levels. When nothing is claimed (a flat
    / near-flat / nuisance-collinear schedule) ``binding_power`` is ``None`` and
    ``identifies_anything`` is ``False`` — never a prior-driven false confidence.
    Returns ``None`` only on degenerate INPUTS.
    """
    names = ["beta", "alpha", "lam"]
    draws = {n: np.asarray(theta_draws.get(n, []), dtype=float) for n in names}
    for n in names:
        draws[n] = draws[n][np.isfinite(draws[n])]
        if draws[n].size == 0:
            return None
    theta_hat = {n: float(np.median(draws[n])) for n in names}
    prior_sd = {n: _robust_sd(draws[n]) for n in names}
    if any(prior_sd[n] <= 0 for n in names):
        return None

    jac = structural_jacobian(
        mults,
        op_spend,
        raw_max,
        y_std,
        theta_hat,
        l_max=l_max,
        normalize=normalize,
        prior_sd=prior_sd,
        fd_rel=fd_rel,
    )
    if jac is None:
        return None
    J = jac["J"]

    sigma_ceiling = max(float(sigma_lo), float(sigma_hi))  # conservative (more noise)
    sigma_floor = min(float(sigma_lo), float(sigma_hi))  # optimistic
    if not math.isfinite(sigma_ceiling) or sigma_ceiling <= 0:
        return None
    info = structural_information(J, names, sigma_ceiling, prior_sd)
    if info is None:
        return None
    info_lo = structural_information(J, names, sigma_floor, prior_sd)

    col_norm = {n: float(np.linalg.norm(J[:, i])) for i, n in enumerate(names)}
    beta_scale = max(col_norm["beta"], _EPS)
    distinct = sorted({round(float(m), 4) for m in np.asarray(mults, float)})
    n_levels = len(distinct)
    sep_ok = n_levels >= 2 and min(np.diff(distinct)) >= _MIN_LEVEL_SEP
    # Structural eligibility — what a design of this SHAPE can even attempt:
    # alpha needs temporal contrast, the saturation curve needs >=3 separated,
    # in-support levels (a binary on/off is a secant, not the curve).
    eligible = {
        "beta": True,
        "alpha": True,
        "lam": n_levels >= 3 and sep_ok and bool(in_support),
    }

    params: dict[str, Any] = {}
    claimed: dict[str, bool] = {}
    for n in names:
        post_sd = info["post_sd"][n]
        post_sd_lo = info_lo["post_sd"][n] if info_lo else float("nan")
        contraction = float(np.clip(1.0 - post_sd / prior_sd[n], 0.0, 1.0))
        # A parameter is "claimed" only when the design is structurally eligible
        # AND the experiment actually MOVES it (absolute contraction floor) — so a
        # flat / near-flat / nuisance-collinear schedule that contributes no
        # information claims nothing and can't drive a falsely-confident headline.
        claimed[n] = bool(
            eligible[n] and math.isfinite(post_sd) and contraction >= _MIN_CLAIM_CONTRACTION
        )
        # power to resolve the parameter from 0 (UI-consistent with ROAS power);
        # reported alongside contraction, which is the honest identification axis.
        power = _power_from_se(post_sd, draws[n])
        power_opt = _power_from_se(post_sd_lo, draws[n]) if info_lo else float("nan")
        identified = bool(claimed[n] and contraction >= _MIN_CONTRACTION)
        params[n] = {
            "claimed": claimed[n],
            "eligible": eligible[n],
            "identified": identified,
            "prior_sd": prior_sd[n],
            "post_sd": post_sd if math.isfinite(post_sd) else None,
            "post_sd_optimistic": post_sd_lo if math.isfinite(post_sd_lo) else None,
            "contraction": contraction,
            "mde": float(_FACTOR * post_sd) if math.isfinite(post_sd) else None,
            "mde_relative": (
                float(_FACTOR * post_sd / abs(theta_hat[n]))
                if math.isfinite(post_sd) and abs(theta_hat[n]) > _EPS
                else None
            ),
            "power": power if math.isfinite(power) else None,
            "power_optimistic": power_opt if math.isfinite(power_opt) else None,
            "theta_hat": theta_hat[n],
            "col_norm_rel": col_norm[n] / beta_scale,
        }

    claimed_names = [n for n in names if claimed[n]]
    powers_claimed = [params[n]["power"] for n in claimed_names]
    contractions_claimed = [params[n]["contraction"] for n in claimed_names]
    # Binding = the worst CLAIMED parameter. None when nothing is claimed (the
    # design identifies nothing — never the prior's resolve-from-0 probability) or
    # when a claimed parameter has a non-finite power.
    binding_power = (
        float(min(powers_claimed))
        if powers_claimed and all(p is not None and math.isfinite(p) for p in powers_claimed)
        else None
    )
    binding_contraction = float(min(contractions_claimed)) if contractions_claimed else None
    return {
        "params": params,
        "names": names,
        "identifies_anything": bool(claimed_names),
        "binding_power": binding_power,
        "binding_contraction": binding_contraction,
        "power_target": float(power_target),
        "n_levels": n_levels,
        "in_support": bool(in_support),
        "lambda_min": info["lambda_min"],
        "condition": info["condition"],
        "equifinality_vec": info["equifinality_vec"],
        "rel_gain_nats": info["rel_gain_nats"],
        # ordered noise bracket (the labels can't invert: floor <= ceiling)
        "sigma_floor": sigma_floor,
        "sigma_ceiling": sigma_ceiling,
        "n_clamped": jac["n_clamped"],
        "upper_bound": True,
    }
