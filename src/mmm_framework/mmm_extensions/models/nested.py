"""
Nested/Mediated MMM implementation.

Models media effects that flow through intermediate mediators
(e.g., awareness, foot traffic) to the final outcome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .base import BaseExtendedMMM
from ..results import MediationEffects

if TYPE_CHECKING:

    from ..config import NestedModelConfig, MediatorConfig


class NestedMMM(BaseExtendedMMM):
    """
    MMM with nested/mediated causal pathways.

    Models: Media → Mediator(s) → Outcome
    With optional direct effects: Media → Outcome

    This model is useful when:
    - You have intermediate metrics (awareness, consideration, traffic)
    - You want to decompose effects into direct vs mediated
    - You have partial observations of mediator variables

    Parameters
    ----------
    X_media : np.ndarray
        Media variable matrix (n_obs, n_channels)
    y : np.ndarray
        Target outcome variable (n_obs,)
    channel_names : list[str]
        Names of media channels
    config : NestedModelConfig
        Configuration specifying mediators and their properties
    mediator_data : dict[str, np.ndarray] | None
        Observed mediator data (where available)
    mediator_masks : dict[str, np.ndarray] | None
        Boolean masks indicating which observations are available
    index : pd.Index | None
        Optional time index
    """

    def __init__(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        config: "NestedModelConfig",
        mediator_data: dict[str, np.ndarray] | None = None,
        mediator_masks: dict[str, np.ndarray] | None = None,
        index: pd.Index | None = None,
        model_config=None,
        trend_config=None,
    ):
        super().__init__(
            X_media,
            y,
            channel_names,
            index,
            model_config=model_config,
            trend_config=trend_config,
        )
        self.config = config
        self.mediator_data = mediator_data or {}
        self.mediator_masks = mediator_masks or {}

        self.mediator_names = [m.name for m in config.mediators]
        self.n_mediators = len(config.mediators)

    def _build_coords(self) -> dict:
        coords = super()._build_coords()
        coords["mediator"] = self.mediator_names
        return coords

    def _get_affecting_channels(self, mediator_name: str) -> list[str]:
        """Get channels that affect a mediator."""
        if mediator_name in self.config.media_to_mediator_map:
            return list(self.config.media_to_mediator_map[mediator_name])
        return self.channel_names

    def _build_mediator_model(
        self,
        med_config: "MediatorConfig",
        media_transformed: pt.TensorVariable,
        channel_mediator_betas: dict,
    ) -> pt.TensorVariable:
        """Build model for a single mediator.

        ``channel_mediator_betas`` is populated in place with this mediator's
        per-channel media-effect coefficient RVs (by reference), keyed
        ``channel_mediator_betas[mediator_name][channel] = beta``.
        """
        from ..config import MediatorType
        from ..components.observation import (
            build_partial_observation_model,
            build_aggregated_survey_observation,
        )
        from ..components.priors import create_effect_prior

        med_name = med_config.name

        # Get affecting channels
        affecting = self._get_affecting_channels(med_name)
        channel_indices = [
            self.channel_names.index(c) for c in affecting if c in self.channel_names
        ]

        # Mediator intercept
        alpha_med = pm.Normal(f"alpha_{med_name}", mu=0, sigma=2)

        # Media → Mediator effects
        constraint = med_config.media_effect.constraint.value

        # Handle single vs multiple channels. Record each channel's media->
        # mediator coefficient RV *by reference* (keyed by the actual channel
        # name, robust to the single- vs multi-channel naming split) so the
        # experiment likelihood can reconstruct the mediated channel effect.
        # The aggregate ``beta_media_to_<med>`` RV is created only on the
        # single-channel path where it is actually used; the multi-channel
        # path creates per-channel ``beta_<channel>_to_<med>`` RVs instead
        # (previously the aggregate RV was created unconditionally and left
        # as a dead, never-informed parameter in the multi-channel graph).
        med_betas: dict[str, pt.TensorVariable] = {}
        if len(channel_indices) == 1:
            idx0 = channel_indices[0]
            beta = create_effect_prior(
                f"beta_media_to_{med_name}",
                constrained=constraint,
                sigma=med_config.media_effect.sigma,
                dims=None,
            )
            media_effect = beta * media_transformed[:, idx0]
            med_betas[self.channel_names[idx0]] = beta
        else:
            # Create per-channel betas
            betas = []
            for idx in channel_indices:
                ch_name = self.channel_names[idx]
                b = create_effect_prior(
                    f"beta_{ch_name}_to_{med_name}",
                    constrained=constraint,
                    sigma=med_config.media_effect.sigma,
                )
                betas.append(b * media_transformed[:, idx])
                med_betas[ch_name] = b
            media_effect = sum(betas)
        channel_mediator_betas[med_name] = med_betas

        # Latent mediator value
        mediator_latent = alpha_med + media_effect

        # Observation model
        if med_config.mediator_type == MediatorType.FULLY_LATENT:
            pass  # No observation

        elif med_config.mediator_type == MediatorType.AGGREGATED_SURVEY:
            # Aggregated survey observation
            build_aggregated_survey_observation(
                name=med_name,
                latent=mediator_latent,
                observed_data=self.mediator_data[med_name],
                config=med_config.aggregated_survey_config,
                is_proportion=True,
            )

        elif med_name in self.mediator_data:
            # Point observation model
            obs_data = self.mediator_data[med_name]
            mask = self.mediator_masks.get(med_name, ~np.isnan(obs_data))

            build_partial_observation_model(
                med_name,
                mediator_latent,
                obs_data,
                mask,
                med_config.observation_noise_sigma,
            )

        # Store for diagnostics
        pm.Deterministic(f"{med_name}_latent", mediator_latent, dims="obs")

        return mediator_latent

    def _build_model(self) -> pm.Model:
        """Build the nested model."""
        from ..components.priors import create_effect_prior

        coords = self._build_coords()

        # Per-(channel) transform handles and media->mediator coefficient RVs,
        # captured locally (never on ``self`` -- they are graph objects) so the
        # experiment likelihood can reconstruct each channel's total (mediated +
        # direct) effect on the outcome.
        channel_mediator_betas: dict[str, dict] = {}
        channel_tx: dict = {}

        with pm.Model(coords=coords) as model:
            # Data. The likelihood operates on the standardized outcome so the
            # fixed effect/noise priors are well-calibrated for any KPI scale;
            # report-consumed deterministics are registered in original units.
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))
            y_standardized = (np.asarray(self.y, dtype=float) - self.y_mean) / (
                self.y_std
            )
            y = pm.Data("y", y_standardized, dims="obs")

            # Media transformations: normalize -> geometric adstock -> logistic
            # saturation (the previously-unused ``alpha`` is now the carryover).
            media_transformed = []
            for i, channel in enumerate(self.channel_names):
                x = X_media[:, i]

                alpha = pm.Beta(f"alpha_{channel}", alpha=2, beta=2)
                lam = pm.Gamma(f"lambda_{channel}", alpha=3, beta=1)

                apply = self._media_transform_apply(i, alpha, lam)
                x_sat = apply(x)
                media_transformed.append(x_sat)
                channel_tx[channel] = (x_sat, apply, x)

            media_transformed = pt.stack(media_transformed, axis=1)

            # Build mediator models
            mediator_values = {}
            for med_config in self.config.mediators:
                mediator_values[med_config.name] = self._build_mediator_model(
                    med_config, media_transformed, channel_mediator_betas
                )

            # Outcome model
            alpha_y = pm.Normal("alpha_y", mu=0, sigma=2)

            # Mediator → Outcome effects
            mediator_contrib = pt.zeros(self.n_obs)
            gammas: dict = {}  # mediator -> outcome-effect RV (by reference)
            for med_config in self.config.mediators:
                med_name = med_config.name

                gamma = create_effect_prior(
                    f"gamma_{med_name}",
                    constrained=med_config.outcome_effect.constraint.value,
                    sigma=med_config.outcome_effect.sigma,
                )
                gammas[med_name] = gamma

                med_effect = gamma * mediator_values[med_name]
                mediator_contrib = mediator_contrib + med_effect

                # Original-unit deviation contribution (reports consume this)
                pm.Deterministic(
                    f"effect_{med_name}_on_y", med_effect * self.y_std, dims="obs"
                )

            # Direct media effects
            direct_contrib = pt.zeros(self.n_obs)
            deltas: dict = {}  # channel -> direct-effect RV (by reference)
            for i, channel in enumerate(self.channel_names):
                # Check if any mediator allows direct effects for this channel
                has_direct = any(
                    m.allow_direct_effect
                    and channel in self._get_affecting_channels(m.name)
                    for m in self.config.mediators
                )

                if has_direct:
                    # Honor the granting mediator's direct-effect prior (sigma +
                    # optional sign constraint). The default reproduces the
                    # historical Normal(0, 0.5) — same distribution + logp (the
                    # mu=0 constant is float32 0.0 here vs the old int8 0, an inert
                    # dtype difference).
                    direct_cfg = next(
                        (
                            m.direct_effect
                            for m in self.config.mediators
                            if m.allow_direct_effect
                            and channel in self._get_affecting_channels(m.name)
                        ),
                        None,
                    )
                    if direct_cfg is not None:
                        delta = create_effect_prior(
                            f"delta_direct_{channel}",
                            constrained=direct_cfg.constraint.value,
                            mu=direct_cfg.mu,
                            sigma=direct_cfg.sigma,
                        )
                    else:  # pragma: no cover - has_direct guarantees a match
                        delta = pm.Normal(f"delta_direct_{channel}", mu=0, sigma=0.5)
                    deltas[channel] = delta
                    direct_contrib = direct_contrib + delta * media_transformed[:, i]
                    # Original-unit deviation contribution
                    pm.Deterministic(
                        f"direct_effect_{channel}",
                        delta * media_transformed[:, i] * self.y_std,
                        dims="obs",
                    )

            # Per-channel TOTAL contribution to the outcome, original KPI units
            # (obs, channel) — the surface the reporting / response-curve tooling
            # reads via ``sample_channel_contributions``. The mediator is linear
            # in the saturated media, so channel c's contribution to the outcome
            # mean is ``(Σ_med gamma_med·beta_{c,med} + delta_c)·sat_c·y_std``:
            # its mediated paths plus its direct path. A pure Deterministic of
            # already-connected RVs — adds no free RV and re-evaluates under a
            # counterfactual ``set_data("X_media")`` for response curves.
            contrib_cols = []
            for i, channel in enumerate(self.channel_names):
                coef = pt.zeros(())
                for med_name, med_map in channel_mediator_betas.items():
                    if channel in med_map:
                        coef = coef + gammas[med_name] * med_map[channel]
                if channel in deltas:
                    coef = coef + deltas[channel]
                contrib_cols.append(coef * media_transformed[:, i])
            pm.Deterministic(
                "channel_contributions",
                pt.stack(contrib_cols, axis=1) * self.y_std,
                dims=("obs", "channel"),
            )

            # Baseline dynamics (standardized scale): a trend and/or Fourier
            # seasonality when the spec provides them, so a real drift or
            # seasonal pattern is not absorbed into the media coefficients.
            # Both terms are None (no RVs added) when unconfigured — so a model
            # built without a model_config/trend_config is byte-identical.
            from ..components.temporal import (
                build_seasonality_contribution,
                build_trend_contribution,
            )
            from ..components.outcome import build_outcome_likelihood

            trend_contrib = build_trend_contribution("", self.n_obs, self.trend_config)
            seasonality_contrib = build_seasonality_contribution(
                "",
                self.index,
                self.n_obs,
                getattr(self.model_config, "seasonality", None),
            )

            # Combine (standardized scale for the likelihood; ``mu`` is
            # registered in original units for reporting/oracles)
            mu_standardized = alpha_y + mediator_contrib + direct_contrib
            if trend_contrib is not None:
                mu_standardized = mu_standardized + trend_contrib
                pm.Deterministic(
                    "trend_component", trend_contrib * self.y_std, dims="obs"
                )
            if seasonality_contrib is not None:
                mu_standardized = mu_standardized + seasonality_contrib
                pm.Deterministic(
                    "seasonality_component",
                    seasonality_contrib * self.y_std,
                    dims="obs",
                )
            pm.Deterministic(
                "mu", mu_standardized * self.y_std + self.y_mean, dims="obs"
            )

            # Likelihood — Normal by default (byte-identical), or the spec's
            # outcome family (e.g. Student-t) on the standardized scale.
            build_outcome_likelihood(
                "y_obs",
                mu_standardized,
                y,
                getattr(self.model_config, "likelihood", None),
                dims="obs",
            )

            # Derived: indirect effects
            self._add_indirect_effect_deterministics(model, media_transformed)

            # Experiment calibration: a channel's total effect on the outcome is
            # its mediated path(s) plus any direct effect --
            # coef_c = sum_m beta_{c->m} * gamma_m + delta_c -- applied to the
            # channel's saturated spend. The coefficient lives on the
            # standardized-outcome scale, so ``y_std`` converts the estimand
            # back to original units.
            if self.experiments:
                self._add_experiment_likelihoods(
                    self._build_experiment_handles(
                        channel_tx, gammas, deltas, channel_mediator_betas
                    ),
                    scale=self.y_std,
                )

        return model

    def _build_experiment_handles(
        self,
        channel_tx: dict,
        gammas: dict,
        deltas: dict,
        channel_mediator_betas: dict,
    ) -> dict:
        """Assemble per-channel experiment handles for the nested model.

        The effective coefficient is the sum over mediators of the channel's
        media->mediator coefficient times that mediator's outcome coefficient,
        plus the channel's direct effect (when present). Channels with no path to
        the outcome are omitted (an experiment on one is skipped with a warning).
        """
        handles: dict = {}
        for channel in self.channel_names:
            coef = None
            for med_name, med_betas in channel_mediator_betas.items():
                if channel in med_betas and med_name in gammas:
                    term = med_betas[channel] * gammas[med_name]
                    coef = term if coef is None else coef + term
            if channel in deltas:
                coef = deltas[channel] if coef is None else coef + deltas[channel]
            if coef is None:
                continue  # channel has no path to the outcome
            x_sat, apply, x_input = channel_tx[channel]
            ch_idx = self.channel_names.index(channel)
            handles[channel] = {
                "coef": coef,
                "x_sat": x_sat,
                "apply": apply,
                "x_input": x_input,
                "spend_obs": self.X_media[:, ch_idx],
            }
        return handles

    def _add_indirect_effect_deterministics(
        self,
        model: pm.Model,
        media_transformed: pt.TensorVariable,
    ):
        """Add derived quantities for effect decomposition."""
        for i, channel in enumerate(self.channel_names):
            for med_config in self.config.mediators:
                med_name = med_config.name

                if channel not in self._get_affecting_channels(med_name):
                    continue

                # Try to get the beta coefficient
                beta_name = f"beta_{channel}_to_{med_name}"
                if beta_name not in model.named_vars:
                    continue

                gamma_name = f"gamma_{med_name}"
                if gamma_name not in model.named_vars:
                    continue

                # Original-unit coefficient: effect on y (KPI units) per unit
                # of saturated spend, matching direct_effect_*/mu scaling
                indirect = model[beta_name] * model[gamma_name] * self.y_std
                pm.Deterministic(f"indirect_{channel}_via_{med_name}", indirect)

    def get_mediation_effects(self) -> pd.DataFrame:
        """Extract mediation effect estimates."""
        self._check_fitted()

        results = []
        posterior = self._trace.posterior

        # ``delta_direct_*`` are free RVs on the standardized-outcome scale;
        # the ``indirect_*`` deterministics are already in original units.
        y_std = getattr(self, "y_std", 1.0)

        for channel in self.channel_names:
            # Direct effect
            direct_var = f"delta_direct_{channel}"
            if direct_var in posterior:
                direct = float(posterior[direct_var].mean()) * y_std
                direct_sd = float(posterior[direct_var].std()) * y_std
            else:
                direct, direct_sd = 0.0, 0.0

            # Indirect effects
            indirect_effects = {}
            total_indirect = 0.0

            for med_config in self.config.mediators:
                med_name = med_config.name
                indirect_var = f"indirect_{channel}_via_{med_name}"

                if indirect_var in posterior:
                    indirect = float(posterior[indirect_var].mean())
                    indirect_effects[med_name] = indirect
                    total_indirect += indirect

            total = direct + total_indirect
            prop_mediated = total_indirect / total if total != 0 else np.nan

            results.append(
                MediationEffects(
                    channel=channel,
                    direct_effect=direct,
                    direct_effect_sd=direct_sd,
                    indirect_effects=indirect_effects,
                    total_indirect=total_indirect,
                    total_effect=total,
                    proportion_mediated=prop_mediated,
                )
            )

        return pd.DataFrame([r.to_dict() for r in results])


__all__ = ["NestedMMM"]
