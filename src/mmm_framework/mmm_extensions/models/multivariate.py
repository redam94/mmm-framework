"""
Multivariate MMM implementation.

Models multiple correlated outcomes with cross-effects
(cannibalization, halo effects) between products/brands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .base import BaseExtendedMMM
from ..results import CrossEffectSummary

if TYPE_CHECKING:
    import arviz as az

    from ..config import MultivariateModelConfig, CrossEffectType


class MultivariateMMM(BaseExtendedMMM):
    """
    MMM with multiple correlated outcomes and cross-effects.

    This model handles:
    - Multiple outcome variables (e.g., sales by product/brand)
    - Correlated residuals using LKJ prior
    - Cross-effects between outcomes (cannibalization, halo)
    - Promotion-modulated cross-effects

    Parameters
    ----------
    X_media : np.ndarray
        Media variable matrix (n_obs, n_channels)
    outcome_data : dict[str, np.ndarray]
        Dictionary mapping outcome names to their data
    channel_names : list[str]
        Names of media channels
    config : MultivariateModelConfig
        Configuration for outcomes and cross-effects
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
        config: "MultivariateModelConfig",
        promotion_data: dict[str, np.ndarray] | None = None,
        index: pd.Index | None = None,
    ):
        # Use first outcome for y (placeholder)
        first_outcome = list(outcome_data.values())[0]
        super().__init__(X_media, first_outcome, channel_names, index)

        self.outcome_data = outcome_data
        self.config = config
        self.promotion_data = promotion_data or {}

        self.outcome_names = [o.name for o in config.outcomes]
        self.n_outcomes = len(config.outcomes)

        # Build cross-effect structure
        self._cross_effect_specs = self._build_cross_effect_specs()

    def _build_coords(self) -> dict:
        coords = super()._build_coords()
        coords["outcome"] = self.outcome_names
        return coords

    def _build_cross_effect_specs(self):
        """Convert config cross-effects to internal specs."""
        from ..components.cross_effects import CrossEffectSpec
        from ..config import CrossEffectType

        specs = []
        for ce in self.config.cross_effects:
            source_idx = self.outcome_names.index(ce.source_outcome)
            target_idx = self.outcome_names.index(ce.target_outcome)

            specs.append(
                CrossEffectSpec(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    effect_type=ce.effect_type.value,
                    prior_sigma=ce.prior_sigma,
                )
            )

            # Add reverse direction for symmetric effects
            if ce.effect_type == CrossEffectType.SYMMETRIC:
                specs.append(
                    CrossEffectSpec(
                        source_idx=target_idx,
                        target_idx=source_idx,
                        effect_type=ce.effect_type.value,
                        prior_sigma=ce.prior_sigma,
                    )
                )

        return specs

    def _build_model(self) -> pm.Model:
        """Build the multivariate model."""
        from ..components.cross_effects import (
            build_cross_effect_matrix,
            compute_cross_effect_contribution,
        )
        from ..components.observation import build_multivariate_likelihood
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

                if self.config.share_media_adstock:
                    if i == 0:
                        alpha_shared = pm.Beta("alpha_shared", alpha=2, beta=2)
                    alpha = alpha_shared
                else:
                    alpha = pm.Beta(f"alpha_{channel}", alpha=2, beta=2)

                lam = pm.Gamma(f"lambda_{channel}", alpha=3, beta=1)
                x_sat = logistic_saturation(x, lam)
                media_transformed.append(x_sat)

            media_transformed = pt.stack(media_transformed, axis=1)

            # Outcome-level parameters
            alpha = pm.Normal("alpha", mu=0, sigma=2, dims="outcome")
            beta_media = pm.Normal(
                "beta_media", mu=0, sigma=0.5, dims=("outcome", "channel")
            )

            # Cross-effects
            if self._cross_effect_specs:
                psi_matrix, psi_params = build_cross_effect_matrix(
                    self._cross_effect_specs,
                    self.n_outcomes,
                    name_prefix="psi",
                )
                pm.Deterministic("psi_matrix", psi_matrix)
            else:
                psi_matrix = None

            # Build expected values
            mu_list = []
            for k, outcome_config in enumerate(self.config.outcomes):
                mu_k = alpha[k]

                # Media effects
                mu_k = mu_k + pt.dot(media_transformed, beta_media[k, :])

                # Cross-effects
                if psi_matrix is not None:
                    # Get promotion modulation
                    modulation = {}
                    for ce in self.config.cross_effects:
                        source_idx = self.outcome_names.index(ce.source_outcome)
                        if ce.promotion_modulated and ce.promotion_column:
                            if ce.promotion_column in self.promotion_data:
                                modulation[source_idx] = pm.Data(
                                    f"promo_{ce.source_outcome}",
                                    self.promotion_data[ce.promotion_column],
                                    dims="obs",
                                )

                    cross_contrib = compute_cross_effect_contribution(
                        Y, psi_matrix, k, self.n_outcomes, modulation
                    )
                    mu_k = mu_k + cross_contrib

                mu_list.append(mu_k)

            mu = pt.stack(mu_list, axis=1)

            # Multivariate likelihood
            build_multivariate_likelihood(
                "Y_obs",
                mu,
                Y_matrix,
                self.n_outcomes,
                self.config.lkj_eta,
                dims=("obs", "outcome"),
            )

        return model

    def get_cross_effects_summary(self) -> pd.DataFrame:
        """Extract cross-effect estimates."""
        import arviz as az

        self._check_fitted()

        if "psi_matrix" not in self._trace.posterior:
            return pd.DataFrame()

        results = []
        psi = self._trace.posterior["psi_matrix"]

        for spec in self._cross_effect_specs:
            vals = psi[:, :, spec.source_idx, spec.target_idx].values.flatten()
            hdi = az.hdi(vals, hdi_prob=0.94)

            results.append(
                CrossEffectSummary(
                    source=self.outcome_names[spec.source_idx],
                    target=self.outcome_names[spec.target_idx],
                    effect_type=spec.effect_type,
                    mean=float(np.mean(vals)),
                    sd=float(np.std(vals)),
                    hdi_low=float(hdi[0]),
                    hdi_high=float(hdi[1]),
                )
            )

        return pd.DataFrame([r.to_dict() for r in results])

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Extract posterior mean correlation matrix."""
        self._check_fitted()

        corr = (
            self._trace.posterior["Y_obs_correlation"]
            .mean(dim=["chain", "draw"])
            .values
        )

        return pd.DataFrame(
            corr,
            index=self.outcome_names,
            columns=self.outcome_names,
        )


__all__ = ["MultivariateMMM"]
