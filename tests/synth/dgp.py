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


def _base_world(seed: int):
    """Shared ingredients for the clean world and most violation scenarios."""
    rng = np.random.default_rng(seed)
    n = N_WEEKS
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


def make_clean(seed: int = 0) -> Scenario:
    """POSITIVE CONTROL: data from the model's exact assumptions."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_unobserved_confounding(seed: int = 1, *, controlled: bool = False) -> Scenario:
    """Latent demand drives BOTH spend (chasing) and baseline sales.

    The classic MMM confounder. The demand-chasing channels (Search/Social) look
    far more effective than they are. With ``controlled=True`` a noisy demand
    proxy is added as a control (closing the back-door) to show the fix.
    """
    rng, weeks, n, _, _, _, _ = _base_world(seed)
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


def make_reverse_causality(seed: int = 2) -> Scenario:
    """Budget pacing: spend chases last period's revenue (simultaneity).

    Spend is set as a fraction of recent sales, so spend and sales are jointly
    determined -- media is not exogenous. Generated forward in time.
    """
    rng, weeks, n, _, _, _, _ = _base_world(seed)
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


def make_multicollinearity(seed: int = 3) -> Scenario:
    """Synchronized flighting: all channels pulse *together* (poor separation).

    Crucially this keeps the same deep pulsing (dark weeks, low-saturation
    regime) as the clean world -- so total media is just as identifiable -- but
    drives every channel from ONE shared burst pattern. The failure is therefore
    attributable to collinearity (can't separate the channels) rather than to any
    loss of overall spend variation.
    """
    rng, weeks, n, _, baseline, controls, _ = _base_world(seed)
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


def make_adstock_misspec(seed: int = 4) -> Scenario:
    """True carryover is long, delayed Weibull with a tail beyond l_max=8."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_saturation_misspec(seed: int = 5) -> Scenario:
    """True saturation is S-shaped (threshold) Hill; model assumes concave 1-exp."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_time_varying_beta(seed: int = 6) -> Scenario:
    """Channel effectiveness drifts + a structural break at the midpoint."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_heavy_tailed_noise(seed: int = 7) -> Scenario:
    """Student-t noise + promo outliers + heteroscedasticity (level-dependent)."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_synergy(seed: int = 8) -> Scenario:
    """Non-additive channels: TV primes Search (multiplicative interaction)."""
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_spend_outliers(seed: int = 9) -> Scenario:
    """Data-entry spikes inflate per-channel max, distorting normalization.

    The model normalizes each channel's spend by its training max. One huge
    erroneous spike per channel sets the max ~15x normal, compressing every real
    week into a tiny near-linear sliver of the saturation curve. The spike is a
    *recording error*: true sales were generated from the un-spiked spend, so the
    correct contribution does not include any lift from the phantom spend.
    """
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_negative_effect(seed: int = 10) -> Scenario:
    """A genuinely negative channel (cannibalization) under a positive-only prior.

    'Display' here is a deep-discount push that pulls demand forward and nets
    NEGATIVE incremental sales. The model's Gamma prior forces every beta >= 0,
    so this truth is structurally unrepresentable.
    """
    rng, weeks, n, spend, baseline, controls, maxes = _base_world(seed)
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


def make_aurora_kitchen_sink(seed: int = 7) -> Scenario:
    """The full Aurora world: confounding + mediation + cannibalization + Hill.

    Demand-blind (the proxy is NOT controlled), so the causal truth is not
    recoverable. Uses the showcase generator's recorded ground truth.
    """
    import sys
    from pathlib import Path

    nbs = str(Path(__file__).resolve().parents[2] / "nbs")
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


def make_realistic(seed: int = 42) -> Scenario:
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
    n = N_WEEKS
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
        spend["Radio"] * (_R_BASE["Print"] / _R_BASE["Radio"]) * rng.lognormal(0, 0.06, n),
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

SCENARIOS: dict[str, Callable[..., Scenario]] = {
    "realistic": make_realistic,
    "clean": make_clean,
    "unobserved_confounding": make_unobserved_confounding,
    "confounding_controlled": lambda seed=1: make_unobserved_confounding(
        seed, controlled=True
    ),
    "reverse_causality": make_reverse_causality,
    "multicollinearity": make_multicollinearity,
    "adstock_misspec": make_adstock_misspec,
    "saturation_misspec": make_saturation_misspec,
    "time_varying_beta": make_time_varying_beta,
    "heavy_tailed_noise": make_heavy_tailed_noise,
    "synergy": make_synergy,
    "spend_outliers": make_spend_outliers,
    "negative_effect": make_negative_effect,
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
    "confounding_controlled",
    "aurora_kitchen_sink",
]


def build(name: str, seed: int | None = None) -> Scenario:
    """Build a scenario by name (uses each factory's default seed if None)."""
    fn = SCENARIOS[name]
    return fn() if seed is None else fn(seed)


__all__ = ["Scenario", "SCENARIOS", "PRIORITY", "build", "CHANNELS"]
