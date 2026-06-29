"""Structural-violation data-generating processes (DGPs) for the MMM.

Every scenario shares one **clean core world** (:func:`_base_world`) whose media
response is built from the model's *exact* structural family:

* geometric carryover with normalized weights (``parametric_adstock`` family),
* concave saturation ``1 - exp(-lambda * x_norm)`` (the model's saturation),
* additive channel contributions,
* a Fourier-representable seasonality and a linear trend,
* time-invariant positive coefficients,
* homoscedastic Gaussian noise,
* **exogenous** spend (independent of the latent demand / outcome).

:func:`make_clean` is the **positive control**: data drawn from precisely the
model's assumptions. On it, recovery error should be ~0 and every diagnostic
green. Every other factory injects exactly **one** violation that real marketing
data exhibits, holding the rest of the world fixed, so the harness can attribute
any degradation to that single broken assumption.

Ground truth is defined the way the *model itself* reports an effect: the
**counterfactual zero-out** of a channel's spend, evaluated on the noiseless
structural mean (:func:`_counterfactual_truth`). This makes "true contribution"
exactly comparable to ``BayesianMMM.compute_counterfactual_contributions`` --
truth and estimate are the same estimand on the same (KPI) scale. For confounded
or endogenous worlds, truth is the **causal** media effect (zeroing spend does
not change the demand-driven baseline), so the gap to the estimate *is* the bias.

A scenario sets ``representable=False`` when the truth lies outside the model's
hypothesis space (e.g. a genuinely negative effect under a positive-only prior).
There, a large error is expected by construction and is reported as such.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNELS = ["TV", "Search", "Social", "Display"]
N_WEEKS = 156  # 3 years weekly -> enough cycles to identify carryover/saturation
START = "2021-01-04"

ResponseFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# numpy transforms (mirror the framework's; kept dependency-free here)
# ---------------------------------------------------------------------------


def _geom_adstock(x: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    """Causal geometric carryover with normalized weights (sum to 1)."""
    if alpha <= 0:
        return x.astype(float)
    w = alpha ** np.arange(l_max, dtype=float)
    w = w / w.sum()
    padded = np.concatenate([np.zeros(l_max - 1), x])
    return np.array([padded[t : t + l_max][::-1] @ w for t in range(len(x))])


def _weibull_adstock(
    x: np.ndarray, shape: float, scale: float, l_max: int
) -> np.ndarray:
    """Causal Weibull-PDF carryover (delayed/humped, long tail), normalized."""
    k = np.arange(l_max, dtype=float)
    w = (
        (shape / scale)
        * (((k + 1) / scale) ** (shape - 1))
        * np.exp(-(((k + 1) / scale) ** shape))
    )
    w = w / w.sum()
    padded = np.concatenate([np.zeros(l_max - 1), x])
    return np.array([padded[t : t + l_max][::-1] @ w for t in range(len(x))])


def _logistic_sat(x_norm: np.ndarray, lam: float) -> np.ndarray:
    """The model's saturation: concave, monotone, bounded in [0, 1)."""
    return 1.0 - np.exp(-lam * x_norm)


def _hill_sat(x_norm: np.ndarray, half: float, hill: float) -> np.ndarray:
    """S-shaped Hill saturation; ``hill > 1`` gives a low-spend threshold."""
    xn = np.clip(x_norm, 0, None)
    return xn**hill / (xn**hill + half**hill)


# ---------------------------------------------------------------------------
# scenario container
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A ready-to-fit world plus the ground truth used to grade the model."""

    name: str
    violates: str  # the model assumption this scenario breaks ("" for control)
    description: str
    weeks: pd.DatetimeIndex
    spend: pd.DataFrame  # observed media spend fed to the model (KPI $000s scale)
    y: pd.Series  # observed KPI
    controls: pd.DataFrame  # control variables (always >= 1 column)
    true_contribution: pd.Series  # per-channel total causal incremental KPI
    true_roas: pd.Series  # per-channel total-effect ROAS (causal)
    representable: bool = True  # is the truth inside the model's hypothesis space?
    control_roles: dict | None = None  # control name -> CausalControlRole
    notes: dict = field(default_factory=dict)

    @property
    def channels(self) -> list[str]:
        return list(self.spend.columns)

    def panel(self):
        """Build a single-KPI :class:`PanelDataset` ready for ``BayesianMMM``."""
        from mmm_framework.config import (
            ControlVariableConfig,
            DimensionType,
            KPIConfig,
            MediaChannelConfig,
            MFFConfig,
        )
        from mmm_framework.data_loader import PanelCoordinates, PanelDataset

        controls = list(self.controls.columns)
        coords = PanelCoordinates(
            periods=self.weeks,
            geographies=None,
            products=None,
            channels=self.channels,
            controls=controls,
        )
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
                for c in self.channels
            ],
            controls=[
                ControlVariableConfig(
                    name=c,
                    dimensions=[DimensionType.PERIOD],
                    causal_role=(self.control_roles or {}).get(c),
                )
                for c in controls
            ],
        )
        return PanelDataset(
            y=pd.Series(self.y.to_numpy(), name="Sales"),
            X_media=self.spend.copy(),
            X_controls=self.controls.copy(),
            coords=coords,
            index=self.weeks,
            config=config,
        )


# ---------------------------------------------------------------------------
# ground truth via the model's own (counterfactual) estimand
# ---------------------------------------------------------------------------


def _counterfactual_truth(
    response_fn: ResponseFn, spend: np.ndarray, channels: list[str]
) -> pd.Series:
    """Per-channel total incremental KPI = full mean - (channel's spend zeroed).

    Evaluated on the *noiseless* structural mean, this is exactly what
    ``compute_counterfactual_contributions`` estimates (set the channel's spend
    column to zero and difference the predictions), so truth and estimate are the
    same estimand. Carryover is removed automatically because zero spend adstocks
    to zero. Works for additive *and* interaction DGPs.
    """
    mu_full = response_fn(spend)
    out = {}
    for i, c in enumerate(channels):
        s0 = spend.copy()
        s0[:, i] = 0.0
        out[c] = float((mu_full - response_fn(s0)).sum())
    return pd.Series(out, name="true_contribution")


def _finish(
    name: str,
    violates: str,
    description: str,
    weeks: pd.DatetimeIndex,
    spend: pd.DataFrame,
    mu_struct: np.ndarray,
    noise: np.ndarray,
    controls: pd.DataFrame,
    response_fn: ResponseFn,
    *,
    representable: bool = True,
    control_roles: dict | None = None,
    notes: dict | None = None,
) -> Scenario:
    """Assemble a Scenario: observed y = structural mean + noise; truth from CF."""
    channels = list(spend.columns)
    truth = _counterfactual_truth(response_fn, spend.to_numpy(float), channels)
    roas = pd.Series({c: truth[c] / spend[c].sum() for c in channels}, name="true_roas")
    y = pd.Series(np.clip(mu_struct + noise, 1.0, None), index=weeks, name="Sales")
    return Scenario(
        name=name,
        violates=violates,
        description=description,
        weeks=weeks,
        spend=spend,
        y=y,
        controls=controls,
        true_contribution=truth,
        true_roas=roas,
        representable=representable,
        control_roles=control_roles,
        notes=notes or {},
    )


# ---------------------------------------------------------------------------
# the clean core world
# ---------------------------------------------------------------------------

# True media response parameters (shared by the clean control and, where not
# explicitly overridden, by the violation scenarios).
_ALPHA = {"TV": 0.6, "Search": 0.2, "Social": 0.4, "Display": 0.5}  # carryover
_LAM = {"TV": 1.6, "Search": 1.8, "Social": 1.7, "Display": 1.5}  # saturation
_AMP = {"TV": 150.0, "Search": 100.0, "Social": 90.0, "Display": 65.0}  # KPI amp
_BASE_SPEND = {"TV": 100.0, "Search": 60.0, "Social": 50.0, "Display": 40.0}
# Distinct short flighting cycles -> strong temporal variation AND low cross-
# correlation, so each channel's effect is well identified in the clean world.
_FLIGHT_PERIOD = {"TV": 9.0, "Search": 7.0, "Social": 11.0, "Display": 13.0}


def _pulsed_levels(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Per-channel multiplicative spend *level* with pulsing (drops near zero).

    Identifiability in an MMM comes from media contribution *varying* over time;
    weeks where a channel is (nearly) dark anchor the contribution level so it is
    separable from the intercept. Pure flat spend would be confounded with the
    baseline -- so every scenario builds spend on top of this pulsed level.
    """
    t = np.arange(n)
    levels = {}
    for c in CHANNELS:
        phase = rng.random() * 2 * np.pi
        burst = np.clip(np.sin(2 * np.pi * t / _FLIGHT_PERIOD[c] + phase), 0, None)
        idio = rng.lognormal(0.0, 0.25, n)
        levels[c] = (0.08 + 1.6 * burst) * idio  # deep pulses ~0.05 .. ~1.7
    return levels


def _exogenous_spend(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Independent, well-varying media spend (no link to demand/outcome)."""
    levels = _pulsed_levels(rng, n)
    spend = {c: np.clip(_BASE_SPEND[c] * levels[c], 0.5, None) for c in CHANNELS}
    return pd.DataFrame(spend, columns=CHANNELS)


def _baseline(rng: np.random.Generator, n: int, price: np.ndarray) -> np.ndarray:
    """Non-media mean: intercept + linear trend + Fourier season + price effect."""
    t = np.arange(n)
    season = 35.0 * np.sin(2 * np.pi * t / 52.0) + 22.0 * np.cos(2 * np.pi * t / 52.0)
    trend = 60.0 * (t / n)
    price_effect = -28.0 * (price - price.mean())
    return 300.0 + trend + season + price_effect


def _media_response_fn(
    alpha: dict, lam: dict, amp: dict, maxes: dict, baseline: np.ndarray
) -> ResponseFn:
    """Build the clean additive media response closure (matches the model)."""

    def fn(spend: np.ndarray) -> np.ndarray:
        mu = baseline.copy()
        for i, c in enumerate(CHANNELS):
            xn = spend[:, i] / maxes[c]
            ad = _geom_adstock(xn, alpha[c])
            mu = mu + amp[c] * _logistic_sat(ad, lam[c])
        return mu

    return fn


def _base_world(seed: int, n_weeks: int | None = None):
    """Shared ingredients for the clean world and most violation scenarios."""
    rng = np.random.default_rng(seed)
    n = int(n_weeks) if n_weeks else N_WEEKS
    weeks = pd.date_range(START, periods=n, freq="W-MON")
    price = (
        12.0
        + 0.5 * np.cos(2 * np.pi * np.arange(n) / 52.0)
        - 0.8 * (rng.random(n) < 0.15)
    )
    spend = _exogenous_spend(rng, n)
    baseline = _baseline(rng, n, price)
    controls = pd.DataFrame({"Price": price})
    maxes = {c: float(spend[c].max()) for c in CHANNELS}
    return rng, weeks, n, spend, baseline, controls, maxes


# ===========================================================================
# scenario factories
# ===========================================================================


def make_clean(seed: int = 0, *, n_weeks: int | None = None) -> Scenario:
    """POSITIVE CONTROL: data from the model's exact assumptions."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "clean",
        "",
        "Model's exact generative family (geometric adstock, 1-exp saturation, "
        "additive, Gaussian, constant betas, exogenous spend).",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"role": "positive control"},
    )


def make_unobserved_confounding(
    seed: int = 1, *, controlled: bool = False, n_weeks: int | None = None
) -> Scenario:
    """Latent demand drives BOTH spend (chasing) and baseline sales.

    The classic MMM confounder. The demand-chasing channels (Search/Social) look
    far more effective than they are. With ``controlled=True`` a noisy demand
    proxy is added as a control (closing the back-door) to show the fix.
    """
    rng, weeks, n, _, _, _, _ = _base_world(seed, n_weeks)
    t = np.arange(n)
    # Latent demand: seasonal + growth + AR(1) persistence (hidden in real life).
    dn = rng.normal(0, 0.18, n)
    for k in range(1, n):
        dn[k] += 0.5 * dn[k - 1]
    demand = 1.0 + 0.35 * np.cos(2 * np.pi * t / 52.0) + 0.45 * (t / n) + dn
    demand_c = demand - demand.mean()

    # Same pulsed (identifiable) base as the clean world, but spend is *also*
    # pushed by latent demand for the chasing channels -> the only difference
    # from clean is the endogeneity.
    chase = {"TV": 0.10, "Search": 1.6, "Social": 1.2, "Display": 0.15}
    levels = _pulsed_levels(rng, n)
    spend = {}
    for c in CHANNELS:
        factor = np.clip(1.0 + chase[c] * demand_c, 0.1, None)
        spend[c] = np.clip(_BASE_SPEND[c] * levels[c] * factor, 0.5, None)
    spend = pd.DataFrame(spend, columns=CHANNELS)
    maxes = {c: float(spend[c].max()) for c in CHANNELS}

    price = 12.0 + 0.5 * np.cos(2 * np.pi * t / 52.0)
    # Baseline is pushed hard by latent demand -- the open back-door.
    baseline = _baseline(rng, n, price) + 90.0 * demand_c
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)

    proxy = 100 * demand + rng.normal(0, 6, n)  # noisy observable proxy
    controls = pd.DataFrame({"Price": price})
    control_roles = None
    if controlled:
        from mmm_framework.config import CausalControlRole

        controls["CategoryDemand"] = proxy
        # Mark the proxy a CONFOUNDER so the model gives it the wide, un-shrunk
        # prior (sigma=2.0) that actually closes the back-door. Added as a plain
        # control it would get the narrow precision prior (sigma=0.5) and be
        # shrunk toward zero -- a half-closed door that leaves residual bias.
        control_roles = {"CategoryDemand": CausalControlRole.CONFOUNDER}
    return _finish(
        "confounding_controlled" if controlled else "unobserved_confounding",
        "no unobserved confounding (exogeneity of spend)",
        "Latent demand drives both spend (Search/Social chase it) and sales; "
        + (
            "demand proxy IS controlled as a CONFOUNDER (back-door closed)."
            if controlled
            else "demand is UNOBSERVED (back-door open)."
        ),
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        control_roles=control_roles,
        notes={"latent_demand": demand, "chasers": ["Search", "Social"]},
    )


def make_reverse_causality(seed: int = 2, *, n_weeks: int | None = None) -> Scenario:
    """Budget pacing: spend chases last period's revenue (simultaneity).

    Spend is set as a fraction of recent sales, so spend and sales are jointly
    determined -- media is not exogenous. Generated forward in time.
    """
    rng, weeks, n, _, _, _, _ = _base_world(seed, n_weeks)
    t = np.arange(n)
    price = 12.0 + 0.5 * np.cos(2 * np.pi * t / 52.0)
    baseline = _baseline(rng, n, price)
    season_demand = 0.30 * np.sin(2 * np.pi * t / 52.0)  # extra demand wobble
    noise = rng.normal(0, 22.0, n)

    pace = {"TV": 0.6, "Search": 1.0, "Social": 0.8, "Display": 0.5}
    levels = _pulsed_levels(rng, n)  # pulsed base keeps each channel identifiable
    spend = np.zeros((n, len(CHANNELS)))
    y = np.zeros(n)
    # Provisional maxes for in-loop normalization (recomputed after).
    prelim_max = {c: _BASE_SPEND[c] * 1.8 for c in CHANNELS}
    y_ref = baseline.mean()
    for k in range(n):
        rev_signal = 0.0 if k == 0 else (y[k - 1] - y_ref) / y_ref
        for i, c in enumerate(CHANNELS):
            sp = _BASE_SPEND[c] * levels[c][k] * (1.0 + pace[c] * rev_signal)
            spend[k, i] = max(0.5, sp)
        # response so far (need carryover -> recompute media up to k cheaply)
        mu_media_k = 0.0
        for i, c in enumerate(CHANNELS):
            xn = spend[: k + 1, i] / prelim_max[c]
            ad = _geom_adstock(xn, _ALPHA[c])[-1]
            mu_media_k += _AMP[c] * _logistic_sat(np.array([ad]), _LAM[c])[0]
        y[k] = max(1.0, baseline[k] + mu_media_k + 22.0 * season_demand[k] + noise[k])

    spend = pd.DataFrame(spend, columns=CHANNELS)
    maxes = {c: float(spend[c].max()) for c in CHANNELS}
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu_struct = fn(spend.to_numpy(float)) + 22.0 * season_demand
    controls = pd.DataFrame({"Price": price})
    return _finish(
        "reverse_causality",
        "no unobserved confounding (exogeneity of spend)",
        "Budgets are paced to recent revenue, so spend and sales are jointly "
        "determined (simultaneity / endogeneity).",
        weeks,
        spend,
        mu_struct,
        noise,
        controls,
        fn,
        notes={"pacing": pace},
    )


def make_multicollinearity(seed: int = 3, *, n_weeks: int | None = None) -> Scenario:
    """Synchronized flighting: all channels pulse *together* (poor separation).

    Crucially this keeps the same deep pulsing (dark weeks, low-saturation
    regime) as the clean world -- so total media is just as identifiable -- but
    drives every channel from ONE shared burst pattern. The failure is therefore
    attributable to collinearity (can't separate the channels) rather than to any
    loss of overall spend variation.
    """
    rng, weeks, n, _, baseline, controls, _ = _base_world(seed, n_weeks)
    t = np.arange(n)
    # ONE shared pulsed driver (same phase/period for every channel).
    phase = rng.random() * 2 * np.pi
    burst = np.clip(np.sin(2 * np.pi * t / 9.0 + phase), 0, None)
    shared = 0.08 + 1.6 * burst
    spend = {}
    for c in CHANNELS:
        idio = rng.lognormal(0.0, 0.05, n)  # almost no independent movement
        spend[c] = np.clip(_BASE_SPEND[c] * shared * idio, 0.5, None)
    spend = pd.DataFrame(spend, columns=CHANNELS)
    maxes = {c: float(spend[c].max()) for c in CHANNELS}
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    corr = float(np.corrcoef(spend.to_numpy().T)[np.triu_indices(4, 1)].mean())
    return _finish(
        "multicollinearity",
        "identifiability (independent spend variation)",
        "All channels share one flighting driver (mean pairwise corr "
        f"~{corr:.2f}); per-channel effects are weakly identified.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"mean_pairwise_corr": corr},
    )


def make_adstock_misspec(seed: int = 4, *, n_weeks: int | None = None) -> Scenario:
    """True carryover is long, delayed Weibull with a tail beyond l_max=8."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    shape = {"TV": 2.6, "Search": 1.8, "Social": 2.2, "Display": 2.4}
    scale = {"TV": 9.0, "Search": 6.0, "Social": 7.0, "Display": 8.0}
    l_true = 26  # carryover well beyond the model's default 8-week window

    def fn(s: np.ndarray) -> np.ndarray:
        mu = baseline.copy()
        for i, c in enumerate(CHANNELS):
            xn = s[:, i] / maxes[c]
            ad = _weibull_adstock(xn, shape[c], scale[c], l_true)
            mu = mu + _AMP[c] * _logistic_sat(ad, _LAM[c])
        return mu

    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "adstock_misspec",
        "adstock functional form / carryover window",
        "True carryover is a delayed Weibull peaking ~6-8 weeks out with mass "
        "past 26 weeks; the model uses geometric carryover truncated at 8 weeks.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"true_l_max": l_true},
    )


def make_saturation_misspec(seed: int = 5, *, n_weeks: int | None = None) -> Scenario:
    """True saturation is S-shaped (threshold) Hill; model assumes concave 1-exp."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    half = {"TV": 0.45, "Search": 0.40, "Social": 0.42, "Display": 0.45}
    hill = {"TV": 3.0, "Search": 2.5, "Social": 3.0, "Display": 2.8}

    def fn(s: np.ndarray) -> np.ndarray:
        mu = baseline.copy()
        for i, c in enumerate(CHANNELS):
            xn = s[:, i] / maxes[c]
            ad = _geom_adstock(xn, _ALPHA[c])
            mu = mu + _AMP[c] * _hill_sat(ad, half[c], hill[c])
        return mu

    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "saturation_misspec",
        "saturation functional form (concavity / no threshold)",
        "True response is S-shaped with a low-spend threshold (Hill coef ~3); "
        "the model assumes a strictly concave 1-exp curve.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"hill": hill},
    )


def make_time_varying_beta(seed: int = 6, *, n_weeks: int | None = None) -> Scenario:
    """Channel effectiveness drifts + a structural break at the midpoint."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    t = np.arange(n)
    brk = n // 2
    # TV fatigues (decays), Search jumps at a break (algo/creative refresh).
    mult = {
        "TV": np.linspace(1.4, 0.6, n),
        "Search": np.where(t < brk, 0.6, 1.5),
        "Social": 1.0 + 0.3 * np.sin(2 * np.pi * t / 52.0),
        "Display": np.ones(n),
    }

    def fn(s: np.ndarray) -> np.ndarray:
        mu = baseline.copy()
        for i, c in enumerate(CHANNELS):
            xn = s[:, i] / maxes[c]
            ad = _geom_adstock(xn, _ALPHA[c])
            mu = mu + _AMP[c] * mult[c] * _logistic_sat(ad, _LAM[c])
        return mu

    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "time_varying_beta",
        "time-invariant coefficients (stationarity)",
        "TV effectiveness fatigues over 2 years and Search jumps at a mid-series "
        "structural break; the model fits one constant beta per channel.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"break_week": int(brk)},
    )


def make_heavy_tailed_noise(seed: int = 7, *, n_weeks: int | None = None) -> Scenario:
    """Student-t noise + promo outliers + heteroscedasticity (level-dependent)."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))
    # Heteroscedastic Student-t (df=3) noise: sd grows with the mean level.
    scale = 14.0 * (mu / mu.mean())
    t_noise = rng.standard_t(3, n) * scale
    # Sparse additive promo spikes (a handful of large positive shocks).
    spikes = np.zeros(n)
    idx = rng.choice(n, size=5, replace=False)
    spikes[idx] = rng.uniform(120, 240, size=5)
    noise = t_noise + spikes
    return _finish(
        "heavy_tailed_noise",
        "Gaussian, homoscedastic likelihood",
        "Noise is heavy-tailed (Student-t, df=3), level-dependent "
        "(heteroscedastic), and contaminated by 5 large promo spikes.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"spike_weeks": sorted(int(i) for i in idx)},
    )


def make_synergy(seed: int = 8, *, n_weeks: int | None = None) -> Scenario:
    """Non-additive channels: TV primes Search (multiplicative interaction)."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    gamma = 130.0  # strength of the TV x Search synergy

    def fn(s: np.ndarray) -> np.ndarray:
        mu = baseline.copy()
        sats = {}
        for i, c in enumerate(CHANNELS):
            xn = s[:, i] / maxes[c]
            ad = _geom_adstock(xn, _ALPHA[c])
            sats[c] = _logistic_sat(ad, _LAM[c])
            mu = mu + _AMP[c] * sats[c]
        mu = mu + gamma * sats["TV"] * sats["Search"]  # interaction term
        return mu

    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "synergy",
        "additive separability of channels",
        "A TV x Search interaction (TV primes search response); the model sums "
        "independent channel effects. Per-channel truths sum to more than total "
        "media because the synergy is credited to both.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"gamma": gamma},
    )


def make_spend_outliers(seed: int = 9, *, n_weeks: int | None = None) -> Scenario:
    """Data-entry spikes inflate per-channel max, distorting normalization.

    The model normalizes each channel's spend by its training max. One huge
    erroneous spike per channel sets the max ~15x normal, compressing every real
    week into a tiny near-linear sliver of the saturation curve. The spike is a
    *recording error*: true sales were generated from the un-spiked spend, so the
    correct contribution does not include any lift from the phantom spend.
    """
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    # Truth is the response to the *true* (un-spiked) spend.
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    sc = Scenario(
        name="spend_outliers",
        violates="robustness of per-channel max-normalization",
        description="One ~15x data-entry spike per channel inflates the "
        "normalization max; real weeks collapse toward zero on the curve.",
        weeks=weeks,
        spend=spend.copy(),  # placeholder; overwritten with spiked spend below
        y=pd.Series(np.clip(mu + noise, 1.0, None), index=weeks, name="Sales"),
        controls=controls,
        true_contribution=_counterfactual_truth(fn, spend.to_numpy(float), CHANNELS),
        true_roas=pd.Series(dtype=float),
        notes={},
    )
    # Inject the erroneous spikes into the OBSERVED spend the model sees.
    spiked = spend.copy()
    spike_weeks = {}
    for c in CHANNELS:
        w = int(rng.integers(10, n - 10))
        spiked.loc[spiked.index[w], c] = float(spend[c].max() * 15.0)
        spike_weeks[c] = w
    sc.spend = spiked
    sc.true_roas = pd.Series(
        {c: sc.true_contribution[c] / spend[c].sum() for c in CHANNELS},
        name="true_roas",
    )  # ROAS on TRUE (un-spiked) spend
    sc.notes = {"spike_weeks": spike_weeks, "true_spend_sum": spend.sum().to_dict()}
    return sc


def make_mixed_data_errors(seed: int = 21, *, n_weeks: int | None = None) -> Scenario:
    """A realistic mix of recording defects — NOT one engineered 15x spike.

    Real pipelines corrupt data in several ways at once, with very different
    magnitudes and detectability:

    * **TV — decimal shift (x10)** on one random *non-dark* week: the classic
      ETL error (value loaded in the wrong unit). Large enough to damage the
      max-normalization; should be detectable.
    * **Social — double-count (x2)** on one random non-dark week: a partial
      re-load. Statistically indistinguishable from a heavy flight week — a
      KNOWN detection limit, kept here so the grading is honest about it.
    * **Search — missed load (recorded 0)** on one non-dark week. Search is
      rebuilt as an ALWAYS-ON channel (real search spend rarely goes dark), so
      a zero week is an anomaly *in context* — but only in context.
    * **Display — untouched**: the within-scenario false-positive control.
    * **KPI — two real promo shocks** (+130..220): genuine demand events that
      should be *modeled* (event dummy), never "corrected" away.

    Error positions are seeded-random anywhere in ``[2, n-3]`` — including
    near the series edges, which window-based detectors handle worst. Truth is
    computed from the TRUE (uncorrupted) spend, like ``make_spend_outliers``.
    """
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)

    # Search becomes always-on: a floor plus mild lognormal variation (no deep
    # pulses) — the realistic "search is never dark" profile.
    spend = spend.copy()
    spend["Search"] = (
        _BASE_SPEND["Search"]
        * (0.8 + 0.4 * rng.random(n))
        * rng.lognormal(0.0, 0.15, n)
    )
    maxes = {c: float(spend[c].max()) for c in CHANNELS}

    # Truth from the TRUE spend (the errors below are recording artifacts).
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))

    def _non_dark_week(col: pd.Series) -> int:
        med = float(col.median())
        while True:
            w = int(rng.integers(2, n - 3))
            if col.iloc[w] > med:
                return w

    errors: dict[str, dict] = {}
    observed = spend.copy()

    w_tv = _non_dark_week(spend["TV"])
    observed.loc[observed.index[w_tv], "TV"] = float(spend["TV"].iloc[w_tv] * 10.0)
    errors["TV"] = {"week": w_tv, "kind": "decimal_shift", "factor": 10.0}

    w_so = _non_dark_week(spend["Social"])
    observed.loc[observed.index[w_so], "Social"] = float(
        spend["Social"].iloc[w_so] * 2.0
    )
    errors["Social"] = {"week": w_so, "kind": "double_count", "factor": 2.0}

    w_se = _non_dark_week(spend["Search"])
    observed.loc[observed.index[w_se], "Search"] = 0.0
    errors["Search"] = {"week": w_se, "kind": "missed_load", "factor": 0.0}

    # Two genuine promo shocks in the KPI (real demand, correctly recorded).
    promo_weeks = sorted(int(w) for w in rng.choice(n, size=2, replace=False))
    promo = np.zeros(n)
    promo[promo_weeks] = rng.uniform(130, 220, size=2)
    noise = rng.normal(0, 22.0, n) + promo

    sc = Scenario(
        name="mixed_data_errors",
        violates="accurate recording of spend & demand events",
        description="Realistic defect mix: a x10 decimal shift (TV), a x2 "
        "double-count (Social, sub-detectable by design), a missed-load zero "
        "(Search, always-on), an untouched channel (Display), and two real KPI "
        "promo shocks that must be modeled, not deleted.",
        weeks=weeks,
        spend=observed,
        y=pd.Series(np.clip(mu + noise, 1.0, None), index=weeks, name="Sales"),
        controls=controls,
        true_contribution=_counterfactual_truth(fn, spend.to_numpy(float), CHANNELS),
        true_roas=pd.Series(dtype=float),
        notes={},
    )
    sc.true_roas = pd.Series(
        {c: sc.true_contribution[c] / spend[c].sum() for c in CHANNELS},
        name="true_roas",
    )
    sc.notes = {
        "errors": errors,
        "promo_weeks": promo_weeks,
        "true_spend_sum": spend.sum().to_dict(),
        "true_search_spend": spend["Search"].copy(),
    }
    return sc


def make_negative_effect(seed: int = 10, *, n_weeks: int | None = None) -> Scenario:
    """A genuinely negative channel (cannibalization) under a positive-only prior.

    'Display' here is a deep-discount push that pulls demand forward and nets
    NEGATIVE incremental sales. The model's Gamma prior forces every beta >= 0,
    so this truth is structurally unrepresentable.
    """
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    amp = dict(_AMP)
    amp["Display"] = -160.0  # genuinely negative incremental

    def fn(s: np.ndarray) -> np.ndarray:
        mu = baseline.copy()
        for i, c in enumerate(CHANNELS):
            xn = s[:, i] / maxes[c]
            ad = _geom_adstock(xn, _ALPHA[c])
            mu = mu + amp[c] * _logistic_sat(ad, _LAM[c])
        return mu

    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "negative_effect",
        "positivity of media effects (Gamma prior, beta >= 0)",
        "'Display' cannibalizes (negative incremental); the positive-only prior "
        "cannot represent a negative coefficient.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        representable=False,
        notes={"negative_channel": "Display"},
    )


def make_trend_break(seed: int = 11, *, n_weeks: int | None = None) -> Scenario:
    """A structural break in the baseline, confounded with a media ramp.

    Mid-series the category takes a level shock (think COVID, a distribution
    loss, a PR crisis) and then recovers along a *new* slope. The brand reacts
    the way real brands do: it ramps TV and Display ~60% to "win back" demand.
    A model with the default LINEAR trend cannot represent the break, and the
    post-break spend ramp is perfectly aligned with the unmodeled structure, so
    the ramped channels absorb the misfit. Measured direction (stress_02): the
    level *shock* dominates the recovery, so TV/Display are *blamed* (TV ~-63%
    under a linear trend) rather than credited -- baseline misfit lands on
    whichever channels' spend moves at the break, in whichever direction
    reconciles the residual.

    The truth is representable: ``TrendType.PIECEWISE`` (or spline/GP) can
    absorb the break — that is the pivot the notebook demonstrates.
    """
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    t = np.arange(n)
    brk = int(n * 0.5)
    # Level drop of ~140 KPI units at the break, then a recovery slope that
    # claws back most (not all) of it by the end of the series.
    shock = np.where(t >= brk, -140.0 + 1.45 * (t - brk), 0.0)
    baseline_brk = baseline + shock

    # The brand's reaction: TV and Display ramp ~60% from the break onward.
    spend = spend.copy()
    ramp = np.where(t >= brk, 1.6, 1.0)
    for c in ("TV", "Display"):
        spend[c] = spend[c] * ramp
    maxes = {c: float(spend[c].max()) for c in CHANNELS}

    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline_brk)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "trend_break",
        "smooth global trend (no structural breaks)",
        "A mid-series level shock with a new recovery slope, while TV/Display "
        "ramp 60% in response; a linear-trend model confounds the recovery "
        "with the ramped media.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={"break_week": brk, "level_shock": -140.0, "recovery_slope": 1.45},
    )


def make_seasonality_misspec(seed: int = 12, *, n_weeks: int | None = None) -> Scenario:
    """Seasonality outside the low-order Fourier family, leaking into media.

    Three real-world departures from the model's order-2 yearly Fourier:

    * the seasonal *amplitude grows* ~60% over the 3 years (the brand scales),
    * sharp **holiday spikes** (two consecutive weeks each late November /
      December) that no low-order Fourier can represent,
    * **Social's flighting is seasonal** — its budget concentrates in Q4, right
      on top of the holiday spikes.

    The danger is targeted: unmodeled holiday lift sits exactly where Social
    spends. Measured outcome: total media stays plausible while the
    per-channel *split* scrambles -- and WHERE the misfit lands is itself
    config-unstable. At the harness's recorded fidelity (PyMC 500x500,
    parametric adstock) Social takes the leak as designed (+116%, uncovered);
    at stress_02's faster settings Social's Q4 saturation leaves no headroom
    and the credit reshuffles to TV (+33%) / Display (-70%) instead. The
    pivots are (a) holiday dummy controls (the indicator is in ``notes``),
    which restores the split (stress_02: med err ~9%), and (b) a higher
    Fourier order, which underfits the spikes and only partially helps.
    """
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed, n_weeks)
    t = np.arange(n)

    # Holiday spikes: weeks 47-48 (Black Friday/Cyber) and 50-51 (Christmas)
    # of each 52-week year. Sharp, additive, non-Fourier-representable.
    week_of_year = t % 52
    holiday = np.isin(week_of_year, [47, 48, 50, 51]).astype(float)

    # Amplitude-growing season replaces the static one from _baseline: rebuild
    # the baseline's non-season parts, then add the evolving season + spikes.
    growth = 1.0 + 0.6 * (t / n)
    season = growth * (
        35.0 * np.sin(2 * np.pi * t / 52.0) + 22.0 * np.cos(2 * np.pi * t / 52.0)
    )
    static_season = 35.0 * np.sin(2 * np.pi * t / 52.0) + 22.0 * np.cos(
        2 * np.pi * t / 52.0
    )
    baseline_sea = baseline - static_season + season + 160.0 * holiday

    # Social's budget concentrates in Q4 (weeks 40-51): seasonal flighting.
    q4 = ((week_of_year >= 40) & (week_of_year <= 51)).astype(float)
    spend = spend.copy()
    spend["Social"] = spend["Social"] * (1.0 + 2.0 * q4)
    maxes = {c: float(spend[c].max()) for c in CHANNELS}

    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline_sea)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)
    return _finish(
        "seasonality_misspec",
        "static low-order Fourier seasonality",
        "Seasonal amplitude grows 60% over 3 years and sharp holiday spikes "
        "(4 weeks/year, +160 KPI) sit exactly where Social concentrates its "
        "Q4 budget; the model's order-2 Fourier can represent none of this.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        notes={
            "holiday_indicator": holiday,
            "q4_indicator": q4,
            "holiday_lift": 160.0,
            "amplitude_growth": 0.6,
            "seasonal_channel": "Social",
        },
    )


def make_dense_controls(seed: int = 13, *, n_weeks: int | None = None) -> Scenario:
    """Many candidate controls on a short series: the variable-selection trap.

    25 candidate controls against n=156 weeks:

    * ``demand_proxy`` — a **confounder**: latent demand drives sales (+180 per
      centered unit) AND the TV/Search budgets chase it. Omit it and those
      channels inherit demand's credit; *shrink* it and the bias comes back.
    * ``price``, ``weather`` — genuine **precision controls** (real effects).
    * 18 pure-noise random walks (true beta = 0) that a wide-prior model will
      happily assign spurious credit on 156 observations.
    * 4 **media-tracking decoys** (``decoy_<channel>``): noisy copies of each
      channel's spend with NO causal effect. A selection prior that keeps them
      hands them the channel's contribution — the bad-control trap inside the
      selection problem.

    ``control_roles`` marks the confounder/precision roles so the model's
    selection machinery can exempt the confounder (never shrink a confounder).
    """
    rng = np.random.default_rng(seed)
    n = int(n_weeks) if n_weeks else N_WEEKS
    weeks = pd.date_range(START, periods=n, freq="W-MON")
    t = np.arange(n)

    # Latent demand (AR(1) + season + growth) -> sales AND TV/Search budgets.
    dn = rng.normal(0, 0.16, n)
    for k in range(1, n):
        dn[k] += 0.5 * dn[k - 1]
    demand = 1.0 + 0.25 * np.cos(2 * np.pi * t / 52.0) + 0.35 * (t / n) + dn
    demand_c = demand - demand.mean()

    chase = {"TV": 2.0, "Search": 2.5, "Social": 0.0, "Display": 0.0}
    levels = _pulsed_levels(rng, n)
    spend = {}
    for c in CHANNELS:
        factor = np.clip(1.0 + chase[c] * demand_c, 0.1, None)
        spend[c] = np.clip(_BASE_SPEND[c] * levels[c] * factor, 0.5, None)
    spend = pd.DataFrame(spend, columns=CHANNELS)
    maxes = {c: float(spend[c].max()) for c in CHANNELS}

    price = 12.0 + 0.5 * np.cos(2 * np.pi * t / 52.0) - 0.8 * (rng.random(n) < 0.15)
    weather = np.sin(2 * np.pi * t / 52.0 + 0.5) + rng.normal(0, 0.3, n)
    season = 35.0 * np.sin(2 * np.pi * t / 52.0) + 22.0 * np.cos(2 * np.pi * t / 52.0)
    baseline = (
        320.0
        + 60.0 * (t / n)
        + season
        + 180.0 * demand_c
        - 26.0 * (price - price.mean())
        + 14.0 * weather
    )

    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 24.0, n)

    controls = {
        "demand_proxy": 100.0 * demand + rng.normal(0, 6.0, n),
        "price": price,
        "weather": weather,
    }
    for i in range(1, 19):
        controls[f"noise_{i}"] = rng.standard_normal(n).cumsum() / np.sqrt(n)
    for c in CHANNELS:  # media-tracking decoys: correlated with spend, beta = 0
        s = spend[c].to_numpy()
        z = (s - s.mean()) / s.std()
        controls[f"decoy_{c.lower()}"] = z + rng.normal(0, 0.45, n)
    controls = pd.DataFrame(controls)

    roles = {
        "demand_proxy": "confounder",
        "price": "precision_control",
        "weather": "precision_control",
    }
    return _finish(
        "dense_controls",
        "a parsimonious, correctly-chosen control set",
        "25 candidate controls on 156 weeks: 1 confounder (demand proxy), 2 "
        "precision controls, 18 noise random walks, and 4 media-tracking "
        "decoys with no causal effect; TV/Search budgets chase the demand the "
        "proxy measures.",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        control_roles=roles,
        notes={
            "true_control_effects": {
                "demand_proxy": "via 180*demand_c (proxy is demand*100 + noise)",
                "price": -26.0,
                "weather": 14.0,
            },
            "noise_controls": [f"noise_{i}" for i in range(1, 19)],
            "decoy_controls": [f"decoy_{c.lower()}" for c in CHANNELS],
            "latent_demand": demand,
        },
    )


def make_aurora_kitchen_sink(seed: int = 7) -> Scenario:
    """The full Aurora world: confounding + mediation + cannibalization + Hill.

    Demand-blind (the proxy is NOT controlled), so the causal truth is not
    recoverable. Uses the showcase generator's recorded ground truth.
    """
    import sys
    from pathlib import Path

    # The aurora generator lives in the repo's nbs/ directory (not shipped
    # with the installed package): search upward from this file for it.
    nbs = next(
        (
            str(p / "nbs")
            for p in Path(__file__).resolve().parents
            if (p / "nbs" / "aurora.py").exists()
        ),
        None,
    )
    if nbs is None:
        raise RuntimeError(
            "aurora_kitchen_sink requires the repo's nbs/aurora.py, which is "
            "not available in this installation."
        )
    if nbs not in sys.path:
        sys.path.insert(0, nbs)
    from aurora import generate_aurora  # type: ignore

    a = generate_aurora(seed=seed)
    return Scenario(
        name="aurora_kitchen_sink",
        violates="multiple (confounding + mediation + non-1-exp saturation)",
        description="Realistic showcase world: latent-demand confounding, "
        "awareness mediation, summer cannibalization, Hill saturation; fit "
        "demand-blind.",
        weeks=a.weeks,
        spend=a.spend.copy(),
        y=pd.Series(a.sales_total, index=a.weeks, name="Sales"),
        controls=pd.DataFrame({"Price": a.price}, index=a.weeks).reset_index(drop=True),
        true_contribution=a.true_contribution.copy(),
        true_roas=a.true_roas.copy(),
        representable=False,
        notes={"source": "nbs/aurora.py", "mediated_share": a.true_mediated_share},
    )


# ---------------------------------------------------------------------------
# a realistic, many-factor world (for the modeling walkthrough)
# ---------------------------------------------------------------------------

# 7 media channels: 2 strong, 2 moderate, 1 moderate-with-mediated-path, 2 weak.
# Adstock/saturation stay in the model's family — the realism is in the *factor
# structure* (confounding, many controls, weak/noisy media), not functional form.
_R_CHANNELS = ["TV", "Search", "Social", "Display", "Video", "Radio", "Print"]
_R_AMP = {
    "TV": 150.0,
    "Search": 130.0,
    "Social": 90.0,
    "Display": 70.0,
    "Video": 55.0,
    # Radio/Print carry REAL, material effects — but they are always bought
    # together on the same flighting calendar (Print's spend is a scaled copy of
    # Radio's; see make_realistic), so they are near-collinear: the model can
    # identify their COMBINED effect but not the split. Each channel's
    # contribution interval stays huge until a lift test pins one of them.
    "Radio": 65.0,
    "Print": 45.0,
}
_R_FLIGHT = {c: 1.6 for c in _R_CHANNELS}  # all channels well-flighted (vary in time)
_R_ALPHA = {
    "TV": 0.6,
    "Search": 0.2,
    "Social": 0.4,
    "Display": 0.5,
    "Video": 0.45,
    "Radio": 0.5,
    "Print": 0.3,
}
_R_LAM = {
    "TV": 1.6,
    "Search": 1.8,
    "Social": 1.7,
    "Display": 1.5,
    "Video": 1.6,
    "Radio": 1.5,
    "Print": 1.4,
}
_R_BASE = {
    "TV": 100.0,
    "Search": 70.0,
    "Social": 55.0,
    "Display": 45.0,
    "Video": 40.0,
    "Radio": 28.0,
    "Print": 18.0,
}
_R_PERIOD = {
    "TV": 9.0,
    "Search": 7.0,
    "Social": 11.0,
    "Display": 13.0,
    "Video": 8.0,
    "Radio": 10.0,
    "Print": 12.0,
}
# How hard each channel's budget chases the (hidden) demand confounder.
_R_CHASE = {
    "TV": 2.0,
    "Search": 3.0,
    "Social": 0.0,
    "Display": 0.0,
    "Video": 0.0,
    "Radio": 0.0,
    "Print": 0.0,
}
_R_WEAK = ["Radio", "Print"]  # tiny effect -> prior-dominated observationally
_R_MEDIATED = ["TV", "Video"]  # also act through brand awareness (a mediator)
_R_CONFOUNDERS = ["category_demand", "distribution"]
_R_PRECISION = ["price", "competitor_promo", "weather", "holiday"]
_R_NOISE = [f"noise_{i}" for i in range(1, 7)]  # 6 irrelevant controls (true β = 0)
_R_MEDIATOR = "brand_awareness"  # post-treatment: media -> awareness -> sales


def make_realistic(seed: int = 42, *, n_weeks: int | None = None) -> Scenario:
    """A realistic many-factor world with noisy, partly-unidentifiable media.

    The challenges a real modeler faces, none of which are functional-form
    misspecification:

    * **Confounding** — a hidden demand signal drives both spend (TV/Search chase
      it) and sales; an observable proxy (``category_demand``) plus
      ``distribution`` are the controls that close the back-door.
    * **Many controls** — 2 confounders, 4 genuine precision controls, **6
      irrelevant noise controls** (true effect 0), and **1 mediator**
      (``brand_awareness``: TV/Video build it, it drives sales — a post-treatment
      variable a naive modeler will wrongly include).
    * **Unidentifiable media** — Radio and Print are always bought together
      (Print's spend is a scaled copy of Radio's), so they are near-collinear:
      observational data pins their *combined* effect but not the split, leaving
      each channel's contribution interval enormous. The model should *say so*
      (a wide credible interval) rather than invent a number; only a lift test
      resolves them.

    Ground truth is the **total causal effect** of each channel (direct **plus**
    the path through awareness), so controlling for the mediator under-counts the
    mediated channels — exactly the bad-control trap.
    """
    rng = np.random.default_rng(seed)
    n = int(n_weeks) if n_weeks else N_WEEKS
    weeks = pd.date_range(START, periods=n, freq="W-MON")
    t = np.arange(n)

    # Hidden demand confounder (AR(1) + season + growth); observed only as a proxy.
    dn = rng.normal(0, 0.16, n)
    for k in range(1, n):
        dn[k] += 0.5 * dn[k - 1]
    demand = 1.0 + 0.30 * np.cos(2 * np.pi * t / 52.0) + 0.40 * (t / n) + dn
    demand_c = demand - demand.mean()

    season = 38.0 * np.sin(2 * np.pi * t / 52.0) + 24.0 * np.cos(2 * np.pi * t / 52.0)
    trend = 70.0 * (t / n)

    # Confounded spend: TV/Search budgets track demand; others are exogenous pulses.
    spend = {}
    for c in _R_CHANNELS:
        phase = rng.random() * 2 * np.pi
        burst = np.clip(np.sin(2 * np.pi * t / _R_PERIOD[c] + phase), 0, None)
        amp = _R_FLIGHT[c]
        idio = rng.lognormal(0, 0.25 if amp > 0.5 else 0.05, n)
        level = ((1.0 - 0.5 * amp) + amp * burst) * idio  # always-on when amp small
        factor = np.clip(1.0 + _R_CHASE[c] * demand_c, 0.1, None)
        spend[c] = np.clip(_R_BASE[c] * level * factor, 0.5, None)
    # Print rides Radio's calendar (a scaled, lightly-noised copy): the two are
    # near-collinear, so observational data can't separate their coefficients.
    spend["Print"] = np.clip(
        spend["Radio"]
        * (_R_BASE["Print"] / _R_BASE["Radio"])
        * rng.lognormal(0, 0.06, n),
        0.5,
        None,
    )
    spend = pd.DataFrame(spend, columns=_R_CHANNELS)
    maxes = {c: float(spend[c].max()) for c in _R_CHANNELS}

    # Observable controls -------------------------------------------------
    price = 12.0 + 0.5 * np.cos(2 * np.pi * t / 52.0) - 0.8 * (rng.random(n) < 0.15)
    competitor = np.clip(
        50.0 + 20.0 * rng.standard_normal(n).cumsum() / np.sqrt(n), 1, None
    )
    weather = np.sin(2 * np.pi * t / 52.0 + 0.5) + rng.normal(0, 0.3, n)
    holiday = (rng.random(n) < 0.08).astype(float)
    distribution = (
        80.0
        + 12.0 * (t / n)
        + 3.0 * np.cos(2 * np.pi * t / 52.0)
        + rng.normal(0, 1.5, n)
    )

    def _direct(sp: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        for i, c in enumerate(_R_CHANNELS):
            xn = sp[:, i] / maxes[c]
            out = out + _R_AMP[c] * _logistic_sat(
                _geom_adstock(xn, _R_ALPHA[c]), _R_LAM[c]
            )
        return out

    def _awareness(sp: np.ndarray) -> np.ndarray:
        # built by TV & Video (their mediated path), 0–100ish scale
        aw = 35.0
        for c in _R_MEDIATED:
            i = _R_CHANNELS.index(c)
            xn = sp[:, i] / maxes[c]
            aw = aw + 30.0 * _logistic_sat(_geom_adstock(xn, _R_ALPHA[c]), _R_LAM[c])
        return aw

    b_med = 4.0  # awareness -> sales rate
    # Fixed (non-media) baseline: confounders + precision controls.
    baseline_fixed = (
        420.0
        + trend
        + season
        + 180.0 * demand_c  # demand confounder -> sales (strong)
        + 4.0 * (distribution - distribution.mean())  # distribution confounder
        - 26.0 * (price - price.mean())  # precision: price (neg)
        - 0.6 * (competitor - competitor.mean())  # precision: competitor (neg)
        + 14.0 * weather  # precision: weather (pos)
        + 30.0 * holiday  # precision: holiday (pos)
    )

    def response_fn(sp: np.ndarray) -> np.ndarray:
        return baseline_fixed + _direct(sp) + b_med * _awareness(sp)

    mu = response_fn(spend.to_numpy(float))
    noise = rng.normal(0, 26.0, n)  # media SNR is realistically low

    # Observed mediator + irrelevant noise controls.
    awareness_obs = np.clip(
        _awareness(spend.to_numpy(float)) + rng.normal(0, 2.5, n), 0, None
    )
    controls = {
        "category_demand": 100.0 * demand + rng.normal(0, 6.0, n),  # noisy proxy
        "distribution": distribution,
        "price": price,
        "competitor_promo": competitor,
        "weather": weather,
        "holiday": holiday,
        _R_MEDIATOR: awareness_obs,
    }
    for nm in _R_NOISE:
        controls[nm] = rng.standard_normal(n).cumsum() / np.sqrt(n)  # random walk, β=0
    controls = pd.DataFrame(controls)

    truth = _counterfactual_truth(response_fn, spend.to_numpy(float), _R_CHANNELS)
    roas = pd.Series(
        {c: truth[c] / spend[c].sum() for c in _R_CHANNELS}, name="true_roas"
    )
    y = pd.Series(np.clip(mu + noise, 1.0, None), index=weeks, name="Sales")

    roles = {c: "confounder" for c in _R_CONFOUNDERS}
    roles.update({c: "precision" for c in _R_PRECISION})
    roles.update({c: "irrelevant" for c in _R_NOISE})
    roles[_R_MEDIATOR] = "mediator"

    return Scenario(
        name="realistic",
        violates="(realistic mix: confounding + many controls + weak/noisy media)",
        description="7 media channels (2 weak/prior-dominated, 2 mediated), 13 controls "
        "(2 confounders, 4 precision, 6 irrelevant, 1 mediator); confounded spend; "
        "low media SNR. Adstock/saturation in the model's family.",
        weeks=weeks,
        spend=spend,
        y=y,
        controls=controls,
        true_contribution=truth,
        true_roas=roas,
        representable=True,
        notes={
            "roles": roles,
            "confounders": _R_CONFOUNDERS,
            "precision": _R_PRECISION,
            "irrelevant": _R_NOISE,
            "mediator": _R_MEDIATOR,
            "weak_channels": _R_WEAK,
            "mediated_channels": _R_MEDIATED,
            "latent_demand": demand,
        },
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def make_economic_health(seed: int = 14, *, n_weeks: int | None = None) -> Scenario:
    """A latent ECONOMIC-HEALTH factor confounds spend and sales — and it is
    *measured* by several noisy indicators.

    The classic unobserved-confounding world (:func:`make_unobserved_confounding`)
    leaves the common cause latent. Here the common cause is **economic health**,
    and it is observed indirectly through a handful of macro indicators
    (GDP growth, consumer confidence, unemployment, retail sales). That is exactly
    the setup a *joint latent-factor MMM* is built for: estimate the latent factor
    from its indicators inside the same graph and condition on it, closing the
    back-door ``spend ← economic health → sales`` that inflates a naive MMM's ROI.

    Ground truth recorded on ``notes`` (the array-valued ones are dropped from the
    JSON answer key but available to a Python test):

    * ``latent_econ`` — the standardized true factor ``E_t`` (shape ``n``),
    * ``indicators`` — ``{name: array}`` of the 4 measured indicators,
    * ``true_loadings`` — the planted loadings (incl. a **negative** one on
      unemployment, to test sign + ordering recovery),
    * ``confound_kappa`` / ``confound_theta`` — the econ→spend / econ→sales
      confounding strengths,
    * ``chasers`` — the channels whose spend chases the economy hardest (where a
      naive MMM is most biased).

    The 4 indicators are NOT placed in ``controls`` (a naive MMM only sees Price):
    they belong to the measurement block and are tagged ``INDICATOR`` by the
    role-tagged dataset the joint model consumes.
    """
    rng, weeks, n, _, _, _, _ = _base_world(seed, n_weeks)
    t = np.arange(n)

    # Latent economic health: AR(1) cycle + slow growth + seasonal wobble. This is
    # the only slow signal in the baseline (besides Price), so the factor is the
    # de-confounder a joint model must recover.
    en = rng.normal(0, 0.18, n)
    for k in range(1, n):
        en[k] += 0.6 * en[k - 1]
    econ = 1.0 + 0.30 * np.cos(2 * np.pi * t / 52.0) + 0.50 * (t / n) + en
    econ_c = (econ - econ.mean()) / (econ.std() + 1e-8)  # standardized truth

    # Indicators loading on econ. gdp_growth is listed FIRST and loads positively:
    # it anchors the factor's orientation for the joint model (whose first loading
    # is sign-pinned). unemployment carries the planted NEGATIVE loading.
    loadings = {
        "gdp_growth": 0.9,
        "consumer_confidence": 0.8,
        "unemployment": -0.7,
        "retail_sales": 0.6,
    }
    psi = 0.5  # idiosyncratic indicator noise (std on the standardized scale)
    indicators = {
        name: lam * econ_c + rng.normal(0, psi, n) for name, lam in loadings.items()
    }

    # econ -> spend: booms drive higher spend; Search/Social chase hardest.
    kappa = {"TV": 0.20, "Search": 1.2, "Social": 1.0, "Display": 0.20}
    levels = _pulsed_levels(rng, n)
    spend = {}
    for c in CHANNELS:
        factor = np.clip(1.0 + kappa[c] * econ_c, 0.1, None)
        spend[c] = np.clip(_BASE_SPEND[c] * levels[c] * factor, 0.5, None)
    spend = pd.DataFrame(spend, columns=CHANNELS)
    maxes = {c: float(spend[c].max()) for c in CHANNELS}

    # econ -> sales: booms raise baseline sales (the open back-door). The baseline
    # is intentionally simple (intercept + price + econ) so the latent factor is
    # the dominant slow signal and is cleanly recoverable.
    price = 12.0 + 0.5 * np.cos(2 * np.pi * t / 52.0)
    price_effect = -28.0 * (price - price.mean())
    theta = 110.0
    baseline = 300.0 + price_effect + theta * econ_c
    fn = _media_response_fn(_ALPHA, _LAM, _AMP, maxes, baseline)
    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 22.0, n)

    controls = pd.DataFrame({"Price": price})
    return _finish(
        "economic_health",
        "no unobserved confounding (exogeneity of spend)",
        "A latent economic-health factor drives both spend (Search/Social chase "
        "it) and baseline sales; it is measured by 4 noisy indicators so a joint "
        "latent-factor MMM can recover and condition on it (closing the back-door "
        "an indicator-free MMM leaves open).",
        weeks,
        spend,
        mu,
        noise,
        controls,
        fn,
        control_roles=None,
        notes={
            "latent_econ": econ_c,
            "indicators": indicators,
            "true_loadings": loadings,
            "confound_kappa": kappa,
            "confound_theta": theta,
            "chasers": ["Search", "Social"],
            "econ_indicators": list(loadings),
        },
    )


# ===========================================================================
# breakout-weighting worlds (one channel split into impression sub-streams)
# ===========================================================================
#
# These three sibling worlds power the falsification harness for the
# breakout-weighting model (``examples/garden_models/breakout_weighted_mmm.py``).
# A single channel (TV) is split into three impression sub-streams that combine
# inside ONE shared saturation curve via a weighted aggregate
# ``Σ_k w_k·I_{k,t}`` — the exact functional form an in-house PSO optimizer
# searches over. The TRUE per-breakout weights are planted with a share-weighted
# mean of 1 (the ``Σ_k w_k·S_k = Σ_k S_k`` sum-preserving constraint), so the
# weights are pure exposure-quality multipliers, not a level change.
#
#   * heterogeneous — weights genuinely differ AND the sub-streams flight
#     independently → the mix is identifiable; a regularized model recovers it.
#   * homogeneous   — weights are all 1 (truth is equal-weighting) → the honest
#     model must COLLAPSE to equal weights (τ→0), where an unregularized
#     optimizer invents spurious weights that still lower in-sample MSE.
#   * collinear     — weights differ (as in heterogeneous) but the sub-streams
#     share ONE flighting calendar → the mix is UNIDENTIFIABLE; the honest model
#     must report WIDE weight posteriors, where the optimizer reports a confident
#     (noise-driven) point.

_BREAKOUT_PARENT = "TV"
_BREAKOUT_SUBS = ["TV_Premium", "TV_Standard", "TV_Remnant"]
_BREAKOUT_PLAIN = ["Search", "Social", "Display"]
# Per-impression effectiveness RATIOS (un-normalized); the makers renormalize to
# a share-weighted mean of 1 against the realized impression totals.
_BREAKOUT_RAW_EFF_HET = {"TV_Premium": 1.8, "TV_Standard": 1.0, "TV_Remnant": 0.4}
_BREAKOUT_RAW_EFF_HOMO = {s: 1.0 for s in _BREAKOUT_SUBS}
# Distinct flighting cycles (weeks) -> independent sub-stream variation when not
# collinear, so the weighted aggregate's *shape* depends on the weights.
_BREAKOUT_SUB_FLIGHT = {"TV_Premium": 6.0, "TV_Standard": 10.0, "TV_Remnant": 15.0}
# Sub-stream base impression levels (sum ~ TV's base spend of 100).
_BREAKOUT_SUB_BASE = {"TV_Premium": 45.0, "TV_Standard": 35.0, "TV_Remnant": 20.0}


def _breakout_substreams(
    rng: np.random.Generator, n: int, *, collinear: bool
) -> dict[str, np.ndarray]:
    """Three TV impression sub-streams.

    ``collinear=False``: each sub-stream has its OWN flighting cycle + idio noise
    (independent variation → the mix is identifiable). ``collinear=True``: one
    shared burst drives all three with almost no independent movement (the
    identifiability ceiling no optimizer can beat).
    """
    t = np.arange(n)
    streams: dict[str, np.ndarray] = {}
    if collinear:
        phase = rng.random() * 2 * np.pi
        burst = np.clip(np.sin(2 * np.pi * t / 9.0 + phase), 0, None)
        shared = 0.08 + 1.6 * burst
        for s in _BREAKOUT_SUBS:
            idio = rng.lognormal(0.0, 0.04, n)  # ~no independent movement
            streams[s] = np.clip(_BREAKOUT_SUB_BASE[s] * shared * idio, 0.5, None)
    else:
        for s in _BREAKOUT_SUBS:
            phase = rng.random() * 2 * np.pi
            burst = np.clip(
                np.sin(2 * np.pi * t / _BREAKOUT_SUB_FLIGHT[s] + phase), 0, None
            )
            idio = rng.lognormal(0.0, 0.25, n)
            streams[s] = np.clip(
                _BREAKOUT_SUB_BASE[s] * (0.08 + 1.6 * burst) * idio, 0.5, None
            )
    return streams


def _make_breakout(
    name: str,
    violates: str,
    description: str,
    *,
    seed: int,
    raw_eff: dict[str, float],
    collinear: bool,
    role: str,
    n_weeks: int | None = None,
) -> Scenario:
    """Assemble a breakout world: TV split into 3 impression sub-streams that feed
    ONE saturation curve via a weighted aggregate with planted, share-mean-1
    weights; Search/Social/Display stay plain channels.

    The channel response is the model's *exact* functional form,
    ``AMP_TV·sat(adstock((Σ_k w*_k·I_k)/M_TV))`` with ``M_TV = max_t Σ_k I_k``
    (the UNWEIGHTED aggregate max), so a correctly-specified breakout model
    recovers ``w*`` up to identifiability.
    """
    rng, weeks, n, base_spend, baseline, controls, _ = _base_world(seed, n_weeks)
    subs = _breakout_substreams(rng, n, collinear=collinear)
    cols = _BREAKOUT_SUBS + _BREAKOUT_PLAIN
    data = {**subs, **{c: base_spend[c].to_numpy(float) for c in _BREAKOUT_PLAIN}}
    spend = pd.DataFrame({c: data[c] for c in cols}, columns=cols)

    # Renormalize the raw effectiveness ratios to a share-weighted mean of 1
    # against the realized impression totals: Σ_k w*_k·S_k = Σ_k S_k exactly.
    S = np.array([float(subs[s].sum()) for s in _BREAKOUT_SUBS])
    w_raw = np.array([float(raw_eff[s]) for s in _BREAKOUT_SUBS])
    w_norm = w_raw * (S.sum() / float(w_raw @ S))
    true_weights = {s: float(w_norm[i]) for i, s in enumerate(_BREAKOUT_SUBS)}

    # Fixed unweighted aggregate max — matches the model's training-time M_C.
    agg = sum(subs[s] for s in _BREAKOUT_SUBS)
    m_tv = float(agg.max())
    plain_max = {c: float(spend[c].max()) for c in _BREAKOUT_PLAIN}

    def fn(sp: np.ndarray) -> np.ndarray:
        mu = baseline.copy()
        a = sp[:, 0] * w_norm[0] + sp[:, 1] * w_norm[1] + sp[:, 2] * w_norm[2]
        mu = mu + _AMP["TV"] * _logistic_sat(
            _geom_adstock(a / m_tv, _ALPHA["TV"]), _LAM["TV"]
        )
        for j, c in enumerate(_BREAKOUT_PLAIN):
            xn = sp[:, 3 + j] / plain_max[c]
            mu = mu + _AMP[c] * _logistic_sat(_geom_adstock(xn, _ALPHA[c]), _LAM[c])
        return mu

    mu = fn(spend.to_numpy(float))
    noise = rng.normal(0, 18.0, n)
    sub_mat = np.array([subs[s] for s in _BREAKOUT_SUBS])
    corr = float(np.corrcoef(sub_mat)[np.triu_indices(3, 1)].mean())
    notes = {
        "breakout_groups": {_BREAKOUT_PARENT: list(_BREAKOUT_SUBS)},
        "true_weights": true_weights,
        "true_raw_effectiveness": dict(raw_eff),
        # Between-breakout log-SD — the τ the partial-pooled model should recover
        # (≈0 when weights are all 1, > 0 when they genuinely differ).
        "true_logtau": float(np.std(np.log(w_norm))),
        "breakout_totals": {s: float(S[i]) for i, s in enumerate(_BREAKOUT_SUBS)},
        "mean_pairwise_corr": corr,
        "unidentifiable": bool(collinear),
        "role": role,
    }
    return _finish(
        name, violates, description, weeks, spend, mu, noise, controls, fn, notes=notes
    )


def make_breakout_heterogeneous(
    seed: int = 30, *, n_weeks: int | None = None
) -> Scenario:
    """TV split into Premium/Standard/Remnant sub-streams with genuinely DIFFERENT
    per-impression effectiveness and INDEPENDENT flighting (identifiable mix).

    The planted weights (share-weighted mean 1) trace a real Premium > Standard >
    Remnant ordering a regularized model can recover.
    """
    return _make_breakout(
        "breakout_heterogeneous",
        "",
        "One channel (TV) split into 3 impression sub-streams with different "
        "per-impression effectiveness and independent flighting; a single "
        "saturation curve over the weighted aggregate. True breakout weights have "
        "share-weighted mean 1 and a recoverable Premium>Standard>Remnant order.",
        seed=seed,
        raw_eff=_BREAKOUT_RAW_EFF_HET,
        collinear=False,
        role="breakout recovery (heterogeneous, identifiable)",
        n_weeks=n_weeks,
    )


def make_breakout_homogeneous(
    seed: int = 31, *, n_weeks: int | None = None
) -> Scenario:
    """Same sub-stream split and distinct flighting, but IDENTICAL per-impression
    effectiveness (true weights all 1).

    The honest test: a partial-pooled model must COLLAPSE to equal-weighting
    (τ→0, every weight's interval covering 1), where an unregularized optimizer
    invents spurious unequal weights that still lower in-sample MSE.
    """
    return _make_breakout(
        "breakout_homogeneous",
        "",
        "TV split into 3 impression sub-streams with distinct flighting but "
        "IDENTICAL per-impression effectiveness (true weights all 1). The honest "
        "model collapses to equal-weighting; an unregularized optimizer overfits "
        "the noise into spurious weights.",
        seed=seed,
        raw_eff=_BREAKOUT_RAW_EFF_HOMO,
        collinear=False,
        role="breakout null (homogeneous -> equal weights)",
        n_weeks=n_weeks,
    )


def make_breakout_collinear(seed: int = 32, *, n_weeks: int | None = None) -> Scenario:
    """Sub-streams differ in true effectiveness (as in heterogeneous) but are
    bought on ONE shared flighting calendar (near-collinear).

    The mix is UNIDENTIFIABLE — only the share-weighted level is pinned, not the
    weight allocation. The honest model must report WIDE weight posteriors
    (τ ≈ its prior), where an optimizer reports a confident, noise-driven point.
    """
    return _make_breakout(
        "breakout_collinear",
        "identifiability (independent breakout variation)",
        "TV sub-streams differ in true effectiveness but share one flighting "
        "calendar (near-collinear); the mix is unidentifiable. The honest model "
        "reports wide weight posteriors rather than a confident point.",
        seed=seed,
        raw_eff=_BREAKOUT_RAW_EFF_HET,
        collinear=True,
        role="breakout unidentifiable (collinear)",
        n_weeks=n_weeks,
    )


SCENARIOS: dict[str, Callable[..., Scenario]] = {
    "realistic": make_realistic,
    "clean": make_clean,
    "unobserved_confounding": make_unobserved_confounding,
    "confounding_controlled": lambda seed=1, **kw: make_unobserved_confounding(
        seed, controlled=True, **kw
    ),
    "reverse_causality": make_reverse_causality,
    "multicollinearity": make_multicollinearity,
    "adstock_misspec": make_adstock_misspec,
    "saturation_misspec": make_saturation_misspec,
    "time_varying_beta": make_time_varying_beta,
    "heavy_tailed_noise": make_heavy_tailed_noise,
    "synergy": make_synergy,
    "spend_outliers": make_spend_outliers,
    "mixed_data_errors": make_mixed_data_errors,
    "negative_effect": make_negative_effect,
    "trend_break": make_trend_break,
    "seasonality_misspec": make_seasonality_misspec,
    "dense_controls": make_dense_controls,
    "economic_health": make_economic_health,
    "breakout_heterogeneous": make_breakout_heterogeneous,
    "breakout_homogeneous": make_breakout_homogeneous,
    "breakout_collinear": make_breakout_collinear,
    "aurora_kitchen_sink": make_aurora_kitchen_sink,
}

# A sensible staged ordering: control first, then high-prevalence silent
# failures, then the rest.
PRIORITY = [
    "clean",
    "unobserved_confounding",
    "reverse_causality",
    "multicollinearity",
    "adstock_misspec",
    "saturation_misspec",
    "time_varying_beta",
    "heavy_tailed_noise",
    "synergy",
    "spend_outliers",
    "negative_effect",
    "trend_break",
    "seasonality_misspec",
    "dense_controls",
    "economic_health",
    "breakout_heterogeneous",
    "breakout_homogeneous",
    "breakout_collinear",
    "confounding_controlled",
    "aurora_kitchen_sink",
]


def build(
    name: str, seed: int | None = None, *, n_weeks: int | None = None
) -> Scenario:
    """Build a scenario by name (uses each factory's default seed if None)."""
    fn = SCENARIOS[name]
    kwargs: dict = {}
    if n_weeks is not None:
        kwargs["n_weeks"] = int(n_weeks)  # aurora_kitchen_sink rejects this
    return fn(**kwargs) if seed is None else fn(seed, **kwargs)


__all__ = ["Scenario", "SCENARIOS", "PRIORITY", "build", "CHANNELS"]
