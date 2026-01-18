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
    ) -> pt.TensorVariable:
        """Build model for a single mediator."""
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

        # Handle single vs multiple channels
        if len(channel_indices) == 1:
            media_effect = beta * media_transformed[:, channel_indices[0]]
        else:
            # Create per-channel betas
            betas = []
            for i, idx in enumerate(channel_indices):
                b = create_effect_prior(
                    f"beta_{affecting[i]}_to_{med_name}",
                    constrained=constraint,
                    sigma=med_config.media_effect.sigma,
                )
                betas.append(b * media_transformed[:, idx])
            media_effect = sum(betas)

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
        from ..components.transforms import logistic_saturation_pt as logistic_saturation

        coords = self._build_coords()

        with pm.Model(coords=coords) as model:
            # Data
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))
            y = pm.Data("y", self.y, dims="obs")

            # Media transformations (simplified)
            media_transformed = []
            for i, channel in enumerate(self.channel_names):
                x = X_media[:, i]

                alpha = pm.Beta(f"alpha_{channel}", alpha=2, beta=2)
                lam = pm.Gamma(f"lambda_{channel}", alpha=3, beta=1)

                x_sat = logistic_saturation(x, lam)
                media_transformed.append(x_sat)

            media_transformed = pt.stack(media_transformed, axis=1)

            # Build mediator models
            mediator_values = {}
            for med_config in self.config.mediators:
                mediator_values[med_config.name] = self._build_mediator_model(
                    med_config, media_transformed
                )

            # Outcome model
            alpha_y = pm.Normal("alpha_y", mu=0, sigma=2)

            # Mediator → Outcome effects
            mediator_contrib = pt.zeros(self.n_obs)
            for med_config in self.config.mediators:
                med_name = med_config.name

                gamma = create_effect_prior(
                    f"gamma_{med_name}",
                    constrained=med_config.outcome_effect.constraint.value,
                    sigma=med_config.outcome_effect.sigma,
                )

                med_effect = gamma * mediator_values[med_name]
                mediator_contrib = mediator_contrib + med_effect

                pm.Deterministic(f"effect_{med_name}_on_y", med_effect, dims="obs")

            # Direct media effects
            direct_contrib = pt.zeros(self.n_obs)
            for i, channel in enumerate(self.channel_names):
                # Check if any mediator allows direct effects for this channel
                has_direct = any(
                    m.allow_direct_effect
                    and channel in self._get_affecting_channels(m.name)
                    for m in self.config.mediators
                )

                if has_direct:
                    delta = pm.Normal(f"delta_direct_{channel}", mu=0, sigma=0.5)
                    direct_contrib = direct_contrib + delta * media_transformed[:, i]
                    pm.Deterministic(
                        f"direct_effect_{channel}",
                        delta * media_transformed[:, i],
                        dims="obs",
                    )

            # Combine
            mu = alpha_y + mediator_contrib + direct_contrib

            # Likelihood
            sigma = pm.HalfNormal("sigma_y", sigma=0.5)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs")

            # Derived: indirect effects
            self._add_indirect_effect_deterministics(model, media_transformed)

        return model

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
