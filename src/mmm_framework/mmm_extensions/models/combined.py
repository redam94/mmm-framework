"""
Combined Nested + Multivariate MMM implementation.

Combines mediated pathways with correlated multi-outcome modeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .base import BaseExtendedMMM

if TYPE_CHECKING:
    import arviz as az

    from ..config import CombinedModelConfig


class CombinedMMM(BaseExtendedMMM):
    """
    Combined nested + multivariate model.

    Supports both mediated pathways AND correlated outcomes with cross-effects.

    This model is useful when:
    - You have multiple outcomes (products/brands)
    - Effects flow through intermediate mediators
    - Outcomes are correlated and may affect each other

    Parameters
    ----------
    X_media : np.ndarray
        Media variable matrix (n_obs, n_channels)
    outcome_data : dict[str, np.ndarray]
        Dictionary mapping outcome names to their data
    channel_names : list[str]
        Names of media channels
    config : CombinedModelConfig
        Configuration for nested + multivariate structure
    mediator_data : dict[str, np.ndarray] | None
        Observed mediator data (where available)
    mediator_masks : dict[str, np.ndarray] | None
        Boolean masks indicating which observations are available
    promotion_data : dict[str, np.ndarray] | None
        Optional promotion indicators for modulated effects
    index : pd.Index | None
        Optional time index
    """

    def __init__(
        self,
        X_media: np.ndarray,
        outcome_data: dict[str, np.ndarray],
        channel_names: list[str],
        config: "CombinedModelConfig",
        mediator_data: dict[str, np.ndarray] | None = None,
        mediator_masks: dict[str, np.ndarray] | None = None,
        promotion_data: dict[str, np.ndarray] | None = None,
        index: pd.Index | None = None,
    ):
        first_outcome = list(outcome_data.values())[0]
        super().__init__(X_media, first_outcome, channel_names, index)

        self.outcome_data = outcome_data
        self.config = config
        self.mediator_data = mediator_data or {}
        self.mediator_masks = mediator_masks or {}
        self.promotion_data = promotion_data or {}

        self.mediator_names = [m.name for m in config.nested.mediators]
        self.outcome_names = [o.name for o in config.multivariate.outcomes]
        self.n_mediators = len(self.mediator_names)
        self.n_outcomes = len(self.outcome_names)

    def _build_coords(self) -> dict:
        coords = super()._build_coords()
        coords["mediator"] = self.mediator_names
        coords["outcome"] = self.outcome_names
        return coords

    def _get_affecting_channels(self, mediator_name: str) -> list[str]:
        """Get channels that affect a mediator."""
        mapping = self.config.nested.media_to_mediator_map
        if mediator_name in mapping:
            return list(mapping[mediator_name])
        return self.channel_names

    def _get_affected_outcomes(self, mediator_name: str) -> list[str]:
        """Get outcomes affected by a mediator."""
        mapping = self.config.mediator_to_outcome_map
        if mediator_name in mapping:
            return list(mapping[mediator_name])
        return self.outcome_names

    def _build_model(self) -> pm.Model:
        """Build the combined model."""
        from ..components.priors import create_effect_prior
        from ..components.observation import (
            build_partial_observation_model,
            build_multivariate_likelihood,
        )
        from ..components.transforms import (
            logistic_saturation_pt as logistic_saturation,
        )

        coords = self._build_coords()

        with pm.Model(coords=coords) as model:
            # Data
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))

            Y_matrix = np.column_stack(
                [self.outcome_data[name] for name in self.outcome_names]
            )
            Y = pm.Data("Y", Y_matrix, dims=("obs", "outcome"))

            # Media transformations
            media_transformed = []
            for i, channel in enumerate(self.channel_names):
                x = X_media[:, i]
                alpha = pm.Beta(f"alpha_{channel}", alpha=2, beta=2)
                lam = pm.Gamma(f"lambda_{channel}", alpha=3, beta=1)
                x_sat = logistic_saturation(x, lam)
                media_transformed.append(x_sat)

            media_transformed = pt.stack(media_transformed, axis=1)

            # Mediator models
            mediator_values = {}
            for med_config in self.config.nested.mediators:
                med_name = med_config.name
                affecting = self._get_affecting_channels(med_name)

                alpha_med = pm.Normal(f"alpha_{med_name}", mu=0, sigma=2)

                # Media effects on mediator
                med_effect = pt.zeros(self.n_obs)
                for ch in affecting:
                    if ch in self.channel_names:
                        ch_idx = self.channel_names.index(ch)
                        constraint = med_config.media_effect.constraint.value
                        beta = create_effect_prior(
                            f"beta_{ch}_to_{med_name}",
                            constrained=constraint,
                            sigma=med_config.media_effect.sigma,
                        )
                        med_effect = med_effect + beta * media_transformed[:, ch_idx]

                mediator_latent = alpha_med + med_effect

                # Observation model
                if med_name in self.mediator_data:
                    obs_data = self.mediator_data[med_name]
                    mask = self.mediator_masks.get(med_name, ~np.isnan(obs_data))
                    build_partial_observation_model(
                        med_name,
                        mediator_latent,
                        obs_data,
                        mask,
                        med_config.observation_noise_sigma,
                    )

                mediator_values[med_name] = mediator_latent
                pm.Deterministic(f"{med_name}_latent", mediator_latent, dims="obs")

            # Outcome parameters
            alpha_y = pm.Normal("alpha_y", mu=0, sigma=2, dims="outcome")
            beta_direct = pm.Normal(
                "beta_direct", mu=0, sigma=0.5, dims=("outcome", "channel")
            )

            # Mediator → Outcome effects
            gamma = pm.Normal("gamma", mu=0, sigma=0.5, dims=("outcome", "mediator"))

            # Cross-effects
            psi = pm.Normal("psi", mu=0, sigma=0.3, dims=("outcome", "outcome"))

            # Build expected values
            mu_list = []
            for k, outcome_config in enumerate(self.config.multivariate.outcomes):
                outcome_name = outcome_config.name
                mu_k = alpha_y[k]

                # Direct media effects
                mu_k = mu_k + pt.dot(media_transformed, beta_direct[k, :])

                # Mediator effects
                for m, med_name in enumerate(self.mediator_names):
                    affected = self._get_affected_outcomes(med_name)
                    if outcome_name in affected:
                        mu_k = mu_k + gamma[k, m] * mediator_values[med_name]

                # Cross-effects
                for j in range(self.n_outcomes):
                    if j != k:
                        mu_k = mu_k + psi[j, k] * Y[:, j]

                mu_list.append(mu_k)

            mu = pt.stack(mu_list, axis=1)

            # Multivariate likelihood
            build_multivariate_likelihood(
                "Y_obs",
                mu,
                Y_matrix,
                self.n_outcomes,
                self.config.multivariate.lkj_eta,
                dims=("obs", "outcome"),
            )

            # Derived quantities
            for i, channel in enumerate(self.channel_names):
                for k, outcome_name in enumerate(self.outcome_names):
                    direct = beta_direct[k, i]

                    indirect = pt.zeros(())
                    for m, med_name in enumerate(self.mediator_names):
                        beta_name = f"beta_{channel}_to_{med_name}"
                        if beta_name in model.named_vars:
                            indirect = indirect + model[beta_name] * gamma[k, m]

                    pm.Deterministic(f"direct_{channel}_{outcome_name}", direct)
                    pm.Deterministic(f"indirect_{channel}_{outcome_name}", indirect)
                    pm.Deterministic(
                        f"total_{channel}_{outcome_name}", direct + indirect
                    )

        return model

    def get_effect_decomposition(self) -> pd.DataFrame:
        """Get full effect decomposition by channel and outcome."""
        self._check_fitted()

        results = []
        posterior = self._trace.posterior

        for channel in self.channel_names:
            for outcome in self.outcome_names:
                direct_var = f"direct_{channel}_{outcome}"
                indirect_var = f"indirect_{channel}_{outcome}"
                total_var = f"total_{channel}_{outcome}"

                row = {"channel": channel, "outcome": outcome}

                for var, name in [
                    (direct_var, "direct"),
                    (indirect_var, "indirect"),
                    (total_var, "total"),
                ]:
                    if var in posterior:
                        row[f"{name}_mean"] = float(posterior[var].mean())
                        row[f"{name}_sd"] = float(posterior[var].std())

                results.append(row)

        return pd.DataFrame(results)


__all__ = ["CombinedMMM"]
