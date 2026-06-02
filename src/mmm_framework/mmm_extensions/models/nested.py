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
    import arviz as az

    from ..config import NestedModelConfig, MediatorConfig, MediatorType


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
    ):
        super().__init__(X_media, y, channel_names, index)
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
        beta = create_effect_prior(
            f"beta_media_to_{med_name}",
            constrained=constraint,
            sigma=med_config.media_effect.sigma,
            dims=None,
        )

        # Handle single vs multiple channels. Record each channel's media->
        # mediator coefficient RV *by reference* (keyed by the actual channel
        # name, robust to the single- vs multi-channel naming split) so the
        # experiment likelihood can reconstruct the mediated channel effect.
        med_betas: dict[str, pt.TensorVariable] = {}
        if len(channel_indices) == 1:
            idx0 = channel_indices[0]
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
            # Data
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))
            y = pm.Data("y", self.y, dims="obs")

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

                pm.Deterministic(f"effect_{med_name}_on_y", med_effect, dims="obs")

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
                    delta = pm.Normal(f"delta_direct_{channel}", mu=0, sigma=0.5)
                    deltas[channel] = delta
                    direct_contrib = direct_contrib + delta * media_transformed[:, i]
                    pm.Deterministic(
                        f"direct_effect_{channel}",
                        delta * media_transformed[:, i],
                        dims="obs",
                    )

            # Combine
            mu = alpha_y + mediator_contrib + direct_contrib
            pm.Deterministic("mu", mu, dims="obs")

            # Likelihood
            sigma = pm.HalfNormal("sigma_y", sigma=0.5)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs")

            # Derived: indirect effects
            self._add_indirect_effect_deterministics(model, media_transformed)

            # Experiment calibration: a channel's total effect on the outcome is
            # its mediated path(s) plus any direct effect --
            # coef_c = sum_m beta_{c->m} * gamma_m + delta_c -- applied to the
            # channel's saturated spend. Single outcome on its raw scale (=1.0).
            if self.experiments:
                self._add_experiment_likelihoods(
                    self._build_experiment_handles(
                        channel_tx, gammas, deltas, channel_mediator_betas
                    ),
                    scale=1.0,
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

                indirect = model[beta_name] * model[gamma_name]
                pm.Deterministic(f"indirect_{channel}_via_{med_name}", indirect)

    def get_mediation_effects(self) -> pd.DataFrame:
        """Extract mediation effect estimates."""
        self._check_fitted()

        results = []
        posterior = self._trace.posterior

        for channel in self.channel_names:
            # Direct effect
            direct_var = f"delta_direct_{channel}"
            if direct_var in posterior:
                direct = float(posterior[direct_var].mean())
                direct_sd = float(posterior[direct_var].std())
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
