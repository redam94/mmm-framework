"""Expected information gain (EIG) for channel experiments, plus the
information-decay model that schedules re-experimentation.

EIG measures how much an experiment of a given precision would shrink the
posterior over a channel's ROI. The Gaussian closed form follows from the
conjugate update: with prior ``N(mu, sigma_k^2)`` and an experiment observing
``y ~ N(roi, sigma_exp^2)``, the posterior variance is
``(1/sigma_k^2 + 1/sigma_exp^2)^-1`` and the entropy reduction is

    EIG = 0.5 * ln(1 + sigma_k^2 / sigma_exp^2)   [nats]

``sigma_exp`` must be DESIGN-GROUNDED (what precision the experiment can
actually achieve, given geo footprint / design type) — if it were derived from
``sigma_k`` itself, every channel would have the same EIG and the ranking
would collapse.

Information decay: experimental evidence has a shelf life. The effective
uncertainty grows as ``sigma_eff^2(t) = sigma_post^2 * exp(lambda * t)`` with
``lambda = ln(2) / half_life``; when the EIG of a fresh experiment (computed
from the decayed sigma) crosses an operational threshold, the channel is due
for a re-test.

Import-light (numpy only) so it can run inside the session kernels.
"""

from __future__ import annotations

import math

import numpy as np

# Achievable measurement sd as a fraction of the channel's median ROI, by
# design type. These are operational defaults (a well-run geo holdout with a
# decent footprint typically lands near ±10% relative precision); override per
# experiment when the footprint is known.
DESIGN_PRECISION: dict[str, float] = {
    "geo_holdout": 0.10,
    "geo_scaling": 0.15,
    "national_pulse": 0.25,
}

# Per channel-CLASS half-lives (weeks) for experimental information: fast-
# moving auction/digital surfaces decay in ~6 months; stable broadcast/brand
# channels hold a year or more.
DEFAULT_HALF_LIVES_WEEKS: dict[str, float] = {
    "search": 26.0,
    "digital": 26.0,
    "display": 26.0,
    "social": 39.0,
    "video": 52.0,
    "tv": 52.0,
    "radio": 52.0,
    "ooh": 52.0,
    "print": 52.0,
    "brand": 52.0,
    "default": 39.0,
}

# EIG (nats) at or above which a fresh experiment is operationally worthwhile.
DEFAULT_RETEST_THRESHOLD_NATS = 0.15

# Freshness floor: evidence younger than this is never flagged for re-test,
# even when the posterior is still wide enough to clear the EIG threshold —
# a channel calibrated last month isn't "stale", it's "recently tested and
# still uncertain" (if anything, the experiment underdelivered; the answer is
# a better design next cycle, not an immediate identical re-run).
MIN_RETEST_AGE_WEEKS = 13.0


def sigma_exp_for_design(
    design_type: str, roi_median: float, *, floor: float = 0.05
) -> float:
    """Design-grounded experiment sd on the ROI scale.

    ``sigma_exp = precision(design) * max(|roi_median|, floor)`` — proportional
    to the channel's ROI scale (a relative-precision design) but NEVER to its
    posterior uncertainty.
    """
    key = str(design_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in DESIGN_PRECISION:
        key = "geo_holdout" if "geo" in key else "national_pulse"
    return DESIGN_PRECISION[key] * max(abs(float(roi_median)), floor)


def eig_gaussian(sigma_k: float, sigma_exp: float) -> float:
    """Closed-form EIG (nats) for a Gaussian prior and Gaussian experiment."""
    if sigma_exp <= 0:
        raise ValueError(f"sigma_exp must be positive, got {sigma_exp}")
    if sigma_k <= 0:
        return 0.0
    return 0.5 * math.log1p((sigma_k / sigma_exp) ** 2)


def _degenerate(x: np.ndarray) -> bool:
    """True when the draws have no meaningful spread (relative to their scale)
    — float dust from identical draws must not feed skew/kurtosis math."""
    return float(x.std()) <= max(float(np.abs(x).max(initial=0.0)), 1.0) * 1e-9


def use_gaussian(roi_draws: np.ndarray) -> bool:
    """Cheap normality gate: closed form unless the posterior is visibly
    skewed or heavy-tailed (|skew| >= 0.5 or excess kurtosis >= 1)."""
    x = np.asarray(roi_draws, dtype=float)
    if x.size < 8 or _degenerate(x):
        return True
    sd = x.std()
    z = (x - x.mean()) / sd
    skew = float(np.mean(z**3))
    ex_kurt = float(np.mean(z**4) - 3.0)
    return abs(skew) < 0.5 and ex_kurt < 1.0


def eig_monte_carlo(
    roi_draws: np.ndarray,
    sigma_exp: float,
    *,
    n_outcomes: int = 64,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo EIG for non-Gaussian ROI posteriors.

    Prior-predictive outcomes ``y_j = roi[d_j] + sigma_exp * z_j``; for each,
    the posterior over draws is the importance reweighting
    ``w_d ∝ N(y_j | roi_d, sigma_exp)`` and entropy is approximated by the
    Gaussian proxy ``0.5 * ln(2*pi*e * Var)`` on the (weighted) draws — exact
    in the Gaussian limit, robust to skew where the closed form overstates.
    """
    if sigma_exp <= 0:
        raise ValueError(f"sigma_exp must be positive, got {sigma_exp}")
    x = np.asarray(roi_draws, dtype=float)
    var0 = float(np.var(x))
    if x.size < 2 or _degenerate(x):
        return 0.0
    rng = rng or np.random.default_rng(0)
    d_idx = rng.integers(0, x.size, size=n_outcomes)
    z = rng.standard_normal(n_outcomes)
    ys = x[d_idx] + sigma_exp * z

    h0 = 0.5 * math.log(2 * math.pi * math.e * var0)
    h_post = np.empty(n_outcomes)
    tiny = var0 * 1e-12
    for j, y in enumerate(ys):
        logw = -0.5 * ((y - x) / sigma_exp) ** 2
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        mu_w = float(w @ x)
        var_w = float(w @ (x - mu_w) ** 2)
        h_post[j] = 0.5 * math.log(2 * math.pi * math.e * max(var_w, tiny))
    return max(0.0, h0 - float(h_post.mean()))


# ── Information decay & re-experimentation ────────────────────────────────────


def channel_half_life(channel: str, overrides: dict[str, float] | None = None) -> float:
    """Half-life (weeks) for a channel, by exact-name override first, then
    keyword match of the channel name against the class table."""
    if overrides and channel in overrides:
        return float(overrides[channel])
    name = channel.lower()
    for key, hl in DEFAULT_HALF_LIVES_WEEKS.items():
        if key != "default" and key in name:
            return hl
    return DEFAULT_HALF_LIVES_WEEKS["default"]


def decayed_sigma(
    sigma_post: float, weeks_elapsed: float, half_life_weeks: float
) -> float:
    """Effective sd after ``weeks_elapsed``:
    ``sigma_eff^2(t) = sigma_post^2 * exp(lambda * t)``, lambda = ln2/half-life
    — so ``sigma_eff(t) = sigma_post * exp(0.5 * lambda * t)``."""
    lam = math.log(2.0) / max(float(half_life_weeks), 1e-9)
    return float(sigma_post) * math.exp(0.5 * lam * max(float(weeks_elapsed), 0.0))


def reexperiment_due(
    sigma_post: float,
    weeks_elapsed: float,
    half_life_weeks: float,
    sigma_exp: float,
    *,
    threshold_nats: float = DEFAULT_RETEST_THRESHOLD_NATS,
    min_age_weeks: float = MIN_RETEST_AGE_WEEKS,
) -> tuple[bool, float]:
    """(due, current_eig): the EIG of a fresh experiment given the decayed
    effective uncertainty, and whether it crosses the operational threshold.
    Evidence younger than ``min_age_weeks`` is never due (freshness floor)."""
    eff = decayed_sigma(sigma_post, weeks_elapsed, half_life_weeks)
    eig = eig_gaussian(eff, sigma_exp)
    due = eig >= threshold_nats and weeks_elapsed >= min_age_weeks
    return due, eig
