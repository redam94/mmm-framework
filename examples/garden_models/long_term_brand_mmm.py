"""Long-term brand estimator — a slow-decaying brand-equity term (issue #122).

A weekly MMM measures *activation* + within-adstock carryover; it does NOT measure
the slow, brand-equity effect that decays over 6–36 months (issue #106 shipped the
estimable within-window split + a caveat + an assumption-driven scenario). This
garden model implements the documented **brand-equity latent / long-term decay**
method (``technical-docs/long-term-brand-effects.md``): each channel's spend feeds
TWO geometric-decay stocks —

* a **fast** activation stock (short half-life — the immediate media effect), and
* a **slow** brand-equity stock (a long, brand-horizon half-life),

and the KPI is base + activation + brand + controls. The split of a channel's
effect into **short-term (activation)** and **long-term (brand)** is then an
ESTIMATE with posterior uncertainty — not the assumption-driven multiplier.

The identification move (the doc's): the two decay HORIZONS are *documented
assumptions* (tight Beta priors on the fast/slow persistence — a brand's memory is
a business input, not something weekly sales can pin), while the two effect
MAGNITUDES (``beta_activation``/``beta_brand``) are *estimated*. This sidesteps
the fast↔slow adstock ridge that would otherwise trade off, so the long-term
contribution is a genuine estimate given an assumed brand-memory horizon. It needs
long history (2–3+ years) to separate the slow stock from the base trend.

Stays ``__garden_model_kind__='mmm'`` (full channel/ROI/report surface). Reuses
the awareness model's scan-free decay-matrix trick. National single-series only.
"""

from __future__ import annotations

import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field

from mmm_framework.estimands.registry import latent_scalar
from mmm_framework.garden import CustomMMM
from mmm_framework.model.base import _apply_saturation_pt, _sample_from_prior_config


class LongTermBrandParams(BaseModel):
    """Config (CONFIG_SCHEMA) for the dual-decay brand model. The two persistence
    priors are the *documented* fast / slow half-lives (tight = a stated business
    assumption); the effect magnitudes are estimated. Half-life ``h`` from a mean
    persistence ``ρ`` is ``ln 2 / -ln ρ`` weeks."""

    #: Fast (activation) persistence prior — short memory (mean ≈ 0.3 → ~0.6 wk).
    fast_retention_alpha: float = Field(2.0, gt=0)
    fast_retention_beta: float = Field(5.0, gt=0)
    #: Slow (brand) persistence prior — long memory (mean ≈ 0.94 → ~11 wk),
    #: TIGHT so the brand horizon is an assumption, not fit from weekly noise.
    slow_retention_alpha: float = Field(47.0, gt=0)
    slow_retention_beta: float = Field(3.0, gt=0)
    #: Prior scale on each channel's brand-effect magnitude (HalfNormal).
    brand_effect_sigma: float = Field(1.0, gt=0)


class LongTermBrandMMM(CustomMMM):
    """MMM with a slow-decaying brand-equity stock alongside fast activation.

    Overrides only ``_build_model`` (Gaussian KPI); everything else — data prep,
    saturation, controls, seasonality, the read-op contract — is the base stack /
    the shared primitives. Registers ``activation_contributions`` /
    ``brand_contributions`` (per channel) and a scalar ``long_term_fraction`` (the
    estimated brand share of total media effect) so the short-vs-long split is a
    first-class, uncertainty-carrying estimand.
    """

    CONFIG_SCHEMA = LongTermBrandParams

    __garden_model_kind__ = "mmm"

    DEFAULT_ESTIMANDS = [
        "contribution_roi",
        "marginal_roas",
        # The headline: the estimated long-term (brand) share of media effect.
        latent_scalar("long_term_brand_share", var="long_term_fraction"),
    ]

    def _build_model(self) -> pm.Model:
        if self.has_geo or self.has_product:
            raise NotImplementedError(
                "LongTermBrandMMM models a single national series (the brand stock "
                "is national). Aggregate to national or use the base BayesianMMM."
            )
        if getattr(self, "experiments", None):
            warnings.warn(
                "LongTermBrandMMM does not implement in-graph experiment "
                "calibration; registered experiments are ignored.",
                stacklevel=2,
            )

        coords = self._build_coords()
        x_media_norm = self._prepare_raw_media_for_model()
        n_obs = self.n_obs
        assert (
            self.time_idx[0] == 0 and self.time_idx[-1] == n_obs - 1
        ), "LongTermBrandMMM expects a single time-ordered national series"
        params = self.model_params  # LongTermBrandParams (defaults applied)

        with pm.Model(coords=coords) as model:
            x_media = pm.Data("X_media_raw", x_media_norm, dims=("obs", "channel"))
            if self.X_controls is not None:
                x_controls = pm.Data(
                    "X_controls", self.X_controls, dims=("obs", "control")
                )
            time_idx = pm.Data("time_idx", self.time_idx)

            intercept = pm.Normal(
                "intercept",
                mu=getattr(self.model_config, "intercept_prior_mu", 0.0),
                sigma=getattr(self.model_config, "intercept_prior_sigma", 0.5),
            )
            pm.Deterministic(
                "intercept_component", intercept + pt.zeros(n_obs), dims="obs"
            )

            # --- the two persistence knobs (documented horizons, tight priors) ---
            rho_fast = pm.Beta(
                "activation_retention",
                alpha=params.fast_retention_alpha,
                beta=params.fast_retention_beta,
            )
            rho_slow = pm.Beta(
                "brand_retention",
                alpha=params.slow_retention_alpha,
                beta=params.slow_retention_beta,
            )

            # --- per-channel saturated inflow + fast/slow magnitudes -------------
            fast_inflow, slow_inflow = [], []
            for c, channel in enumerate(self.channel_names):
                sat_kind, sat_params = self._build_channel_saturation(channel)
                x_saturated = _apply_saturation_pt(x_media[:, c], sat_kind, sat_params)

                media_cfg = self.mff_config.get_media_config(channel)
                roi_prior = getattr(media_cfg, "roi_prior", None)
                beta_act = _sample_from_prior_config(
                    f"beta_{channel}",
                    roi_prior,
                    lambda ch=channel: pm.Gamma(f"beta_{ch}", mu=1.5, sigma=1.0),
                )
                beta_brand = pm.HalfNormal(
                    f"beta_brand_{channel}", sigma=params.brand_effect_sigma
                )
                fast_inflow.append(beta_act * x_saturated)
                slow_inflow.append(beta_brand * x_saturated)
                # adstock reporting reads the (fast) activation persistence.
                pm.Deterministic(f"adstock_alpha_{channel}", rho_fast)

            # --- vectorized geometric-decay stocks (fast + slow) ----------------
            t = np.arange(n_obs)
            lag = t[:, None] - t[None, :]
            causal = pt.as_tensor_variable(lag >= 0)
            lag_clamped = pt.as_tensor_variable(np.maximum(lag, 0))
            decay_fast = pt.where(causal, rho_fast**lag_clamped, 0.0)
            decay_slow = pt.where(causal, rho_slow**lag_clamped, 0.0)

            fast_in = pt.stack(fast_inflow, axis=1)  # (obs, ch)
            slow_in = pt.stack(slow_inflow, axis=1)
            activation = decay_fast @ fast_in  # (obs, ch)
            brand = decay_slow @ slow_in  # (obs, ch)

            pm.Deterministic(
                "activation_contributions", activation, dims=("obs", "channel")
            )
            pm.Deterministic("brand_contributions", brand, dims=("obs", "channel"))
            # channel_contributions = total media effect (the read-op contract).
            channel_contributions = pm.Deterministic(
                "channel_contributions", activation + brand, dims=("obs", "channel")
            )
            media_total = channel_contributions.sum(axis=1)
            pm.Deterministic("media_total", media_total)

            act_total = activation.sum()
            brand_total = brand.sum()
            pm.Deterministic("activation_total", act_total)
            pm.Deterministic("brand_total", brand_total)
            # The estimated long-term (brand) share of the total media effect —
            # a scalar in [0, 1] with full posterior uncertainty. THE #122 number.
            pm.Deterministic(
                "long_term_fraction",
                brand_total / (act_total + brand_total + 1e-9),
            )

            # --- trend / seasonality / controls (reused base machinery) ---------
            # NB no free trend: a flexible trend would compete with the slow brand
            # stock for the low-frequency signal (both are slow), collapsing the
            # long-term estimate. The brand stock IS the slow structural component.
            trend_c = pt.zeros(n_obs)
            pm.Deterministic("trend_component", trend_c, dims="obs")

            seasonality_at_periods = pt.zeros(self.n_periods)
            for name, features in self.seasonality_features.items():
                season_coef = pm.Normal(
                    f"season_{name}",
                    mu=0,
                    sigma=self.seasonality_config.prior_sigma_for(name),
                    shape=features.shape[1],
                    dims=f"{name}_fourier",
                )
                seasonality_at_periods = seasonality_at_periods + pt.dot(
                    pt.as_tensor_variable(features), season_coef
                )
            seasonality = seasonality_at_periods[time_idx]
            pm.Deterministic("seasonality_component", seasonality)
            pm.Deterministic("seasonality_by_period", seasonality_at_periods)

            sigma = (
                pm.HalfNormal("sigma", sigma=0.5) if self._selection_active() else None
            )
            if self.n_controls > 0:
                beta_controls = self._build_control_betas(sigma)
                control_contribution = pt.dot(x_controls, beta_controls)
                pm.Deterministic(
                    "control_contributions",
                    x_controls * beta_controls,
                    dims=("obs", "control"),
                )
            else:
                control_contribution = pt.zeros(n_obs)
            pm.Deterministic("controls_total", control_contribution, dims="obs")

            mu = intercept + trend_c + seasonality + media_total + control_contribution
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=0.5)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y, dims="obs")
            pm.Deterministic(
                "y_obs_scaled", y_obs * self.y_std + self.y_mean, dims="obs"
            )

        return model


# Disambiguate for the Model Garden loader.
GARDEN_MODEL = LongTermBrandMMM
