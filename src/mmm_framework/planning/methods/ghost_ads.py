"""Ghost ads — user-level incrementality power calculator.

Randomization is at the **user/impression** level via the ad server: treated
users see the real ad; "ghost" (or PSA) control users are eligible and
would-have-been-served but see a placebo. The estimand is a conversion-rate (or
per-user count / revenue) lift, so power is a two-proportion / two-mean problem
driven by the baseline rate, users reached, and the treated/ghost split — **no
time series and no fitted MMM required**. This makes the calculator a clean,
standalone pre-fit tool.

Three outcomes:

* ``binary`` — conversion yes/no per user (two-proportion z-test).
* ``count`` — purchases per user (quasi-Poisson: ``var = mean * dispersion``).
* ``revenue`` — mean value per user (two-mean t/z with a supplied per-user SD).

Ghost-ads dilution: only ``exposure_rate`` of randomized users are actually
reached by the campaign, so the intent-to-treat (ITT) effect is
``exposure_rate x`` the treatment-on-treated (TOT) effect. The calculator
reports the MDE on BOTH scales — the experiment *measures* ITT; the business
usually *cares about* TOT.

The normal approximation is optimistic in the rare-event regime (small
``baseline_rate x users``); :func:`ghost_ads_simulate` draws binomial outcomes
to give the empirical power / false-positive rate, and
:func:`ghost_ads_power` flags the regime where the approximation is shaky.

numpy only (kernel-safe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_Z = {  # standard normal quantiles
    0.80: 0.8416212335729143,
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
    0.975: 1.959963984540054,
}


def _z(q: float) -> float:
    if q in _Z:
        return _Z[q]
    # Acklam-style inverse normal via numpy (avoid a scipy dependency here)
    from numpy import sqrt

    # use the erfinv identity: Phi^-1(q) = sqrt(2) * erfinv(2q - 1)
    try:
        from scipy.special import erfinv  # scipy is already a planning dep

        return float(sqrt(2.0) * erfinv(2.0 * q - 1.0))
    except Exception:  # pragma: no cover - scipy always present
        raise ValueError(f"unsupported quantile {q}")


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class GhostAdsDesign:
    """Inputs for the ghost-ads power calculation.

    ``baseline_rate`` is the control conversion probability for ``binary``;
    ``baseline_mean``/``baseline_dispersion`` describe a ``count`` outcome
    (per-user mean and quasi-Poisson dispersion ``var = mean * dispersion``);
    ``baseline_mean``/``value_sd`` describe a ``revenue`` outcome.
    """

    users_reached: int
    baseline_rate: float = 0.02
    treated_fraction: float = 0.5
    outcome: str = "binary"  # 'binary' | 'count' | 'revenue'
    baseline_mean: float | None = None
    baseline_dispersion: float = 1.0
    value_sd: float | None = None
    alpha: float = 0.05
    power_target: float = 0.80
    two_sided: bool = True
    exposure_rate: float = 1.0  # share of randomized users actually reached
    cost_per_user: float | None = None
    value_per_conversion: float | None = None

    def __post_init__(self) -> None:
        if self.users_reached <= 0:
            raise ValueError("users_reached must be positive")
        if not 0.0 < self.treated_fraction < 1.0:
            raise ValueError("treated_fraction must be in (0, 1)")
        if self.outcome == "binary" and not 0.0 < self.baseline_rate < 1.0:
            raise ValueError("baseline_rate must be in (0, 1)")
        if self.outcome in ("count", "revenue") and not self.baseline_mean:
            raise ValueError(f"{self.outcome} outcome needs baseline_mean")
        if self.outcome == "revenue" and not self.value_sd:
            raise ValueError("revenue outcome needs value_sd")
        if not 0.0 < self.exposure_rate <= 1.0:
            raise ValueError("exposure_rate must be in (0, 1]")


def _arm_sizes(design: GhostAdsDesign) -> tuple[float, float]:
    n_t = design.users_reached * design.treated_fraction
    n_c = design.users_reached * (1.0 - design.treated_fraction)
    return n_t, n_c


def _null_variance_per_user(design: GhostAdsDesign) -> float:
    """Per-user outcome variance under the null (control) distribution."""
    if design.outcome == "binary":
        return design.baseline_rate * (1.0 - design.baseline_rate)
    if design.outcome == "count":
        return design.baseline_mean * design.baseline_dispersion
    return float(design.value_sd) ** 2  # revenue


def _se_null(design: GhostAdsDesign) -> float:
    """SE of the treated-minus-control mean difference under the null."""
    n_t, n_c = _arm_sizes(design)
    v = _null_variance_per_user(design)
    return math.sqrt(v / n_t + v / n_c)


def _z_alpha(design: GhostAdsDesign) -> float:
    return _z(1.0 - design.alpha / 2.0) if design.two_sided else _z(1.0 - design.alpha)


def ghost_ads_power(design: GhostAdsDesign) -> dict:
    """Closed-form MDE + diagnostics for a ghost-ads test.

    The MDE solves ``delta = (z_alpha + z_power) * se(delta)`` — for binary
    outcomes the SE depends on the treated rate ``p0 + delta``, so a short
    fixed-point iteration replaces the naive null-SE formula (which understates
    the MDE for lifts that move the rate appreciably).
    """
    z_a = _z_alpha(design)
    z_b = _z(design.power_target)
    n_t, n_c = _arm_sizes(design)

    if design.outcome == "binary":
        p0 = design.baseline_rate
        q0 = 1.0 - p0
        delta = (z_a + z_b) * math.sqrt(p0 * q0 * (1.0 / n_t + 1.0 / n_c))
        for _ in range(50):  # fixed point: se uses p1 = p0 + delta
            p1 = min(p0 + delta, 1.0 - 1e-12)
            se = math.sqrt(p0 * q0 / n_c + p1 * (1.0 - p1) / n_t)
            new = (z_a + z_b) * se
            if abs(new - delta) < 1e-14:
                delta = new
                break
            delta = new
        baseline = p0
    else:
        se = _se_null(design)
        delta = (z_a + z_b) * se
        baseline = float(design.baseline_mean)

    se_null = _se_null(design)
    mde_abs = float(delta)
    mde_rel = float(delta / baseline) if baseline else float("nan")
    incremental = float(delta * n_t)  # incremental conversions/value at the MDE

    # ITT vs TOT: the experiment measures ITT; dilution deflates the effect by
    # exposure_rate, so the TOT MDE is larger by 1/exposure_rate.
    itt_mde = mde_abs
    tot_mde = mde_abs / design.exposure_rate

    # rare-event flag: normal approx needs enough expected events per arm
    expected_events = (
        min(n_t, n_c) * design.baseline_rate if design.outcome == "binary" else None
    )
    rare_event = bool(expected_events is not None and expected_events < 30)

    out = {
        "outcome": design.outcome,
        "users_reached": int(design.users_reached),
        "n_treated": float(n_t),
        "n_ghost": float(n_c),
        "baseline": baseline,
        "se_null": float(se_null),
        "mde_abs": mde_abs,
        "mde_rel": mde_rel,
        "itt_mde": float(itt_mde),
        "tot_mde": float(tot_mde),
        "exposure_rate": design.exposure_rate,
        "incremental_at_mde": incremental,
        "alpha": design.alpha,
        "power_target": design.power_target,
        "two_sided": design.two_sided,
        "rare_event_regime": rare_event,
        "method": "ghost_ads",
    }
    if design.cost_per_user is not None:
        out["media_cost"] = float(design.cost_per_user * n_t)
    if design.value_per_conversion is not None and design.outcome == "binary":
        out["incremental_value_at_mde"] = float(
            incremental * design.value_per_conversion
        )
        if design.cost_per_user is not None:
            cost = design.cost_per_user * n_t
            out["breakeven_lift_abs"] = float(
                cost / (design.value_per_conversion * n_t)
            )
    return out


def ghost_ads_power_at(design: GhostAdsDesign, lift_abs: float) -> float:
    """Two-sided power to detect an absolute lift of ``lift_abs``."""
    z_a = _z_alpha(design)
    se = _se_null(design)
    x = abs(lift_abs) / max(se, 1e-300)
    if design.two_sided:
        return float(_phi(x - z_a) + _phi(-x - z_a))
    return float(_phi(x - z_a))


def ghost_ads_users_for_mde(design: GhostAdsDesign, target_lift_abs: float) -> int:
    """Users required for ``target_lift_abs`` to be the MDE (closed form).

    ``n = ((z_a + z_b)/delta)^2 * (v0/(1-f) + v1/f)`` with per-user variances
    under control/treated outcomes.
    """
    if target_lift_abs <= 0:
        raise ValueError("target_lift_abs must be positive")
    z_a = _z_alpha(design)
    z_b = _z(design.power_target)
    f = design.treated_fraction
    if design.outcome == "binary":
        p0 = design.baseline_rate
        p1 = min(p0 + target_lift_abs, 1.0 - 1e-12)
        v0 = p0 * (1.0 - p0)
        v1 = p1 * (1.0 - p1)
    else:
        v0 = v1 = _null_variance_per_user(design)
    n = ((z_a + z_b) / target_lift_abs) ** 2 * (v0 / (1.0 - f) + v1 / f)
    return int(math.ceil(n))


def ghost_ads_simulate(
    design: GhostAdsDesign,
    true_lift_abs: float,
    *,
    n_sims: int = 2000,
    seed: int = 0,
) -> dict:
    """Empirical power (at ``true_lift_abs``) and FPR (at lift 0) by simulation.

    Validates the normal approximation — particularly in the rare-event regime
    where it is optimistic. ``binary`` outcomes only (counts/revenue are far
    from the boundary in practice).
    """
    if design.outcome != "binary":
        raise ValueError("simulation supports binary outcomes only")
    rng = np.random.default_rng(seed)
    n_t, n_c = (int(round(v)) for v in _arm_sizes(design))
    p0 = design.baseline_rate
    z_a = _z_alpha(design)

    def _reject(p_treat: float) -> float:
        x_t = rng.binomial(n_t, p_treat, size=n_sims)
        x_c = rng.binomial(n_c, p0, size=n_sims)
        ph_t = x_t / n_t
        ph_c = x_c / n_c
        pool = (x_t + x_c) / (n_t + n_c)
        se = np.sqrt(np.clip(pool * (1 - pool), 1e-300, None) * (1 / n_t + 1 / n_c))
        z = (ph_t - ph_c) / se
        return float(np.mean(np.abs(z) > z_a) if design.two_sided else np.mean(z > z_a))

    fpr = _reject(p0)
    power = _reject(min(p0 + true_lift_abs, 1.0 - 1e-12))
    return {
        "empirical_power": power,
        "empirical_fpr": fpr,
        "analytic_power": ghost_ads_power_at(design, true_lift_abs),
        "n_sims": int(n_sims),
        "true_lift_abs": float(true_lift_abs),
    }


__all__ = [
    "GhostAdsDesign",
    "ghost_ads_power",
    "ghost_ads_power_at",
    "ghost_ads_users_for_mde",
    "ghost_ads_simulate",
]
