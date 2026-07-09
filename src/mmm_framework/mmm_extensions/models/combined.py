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
from ..results import CrossEffectSummary

if TYPE_CHECKING:
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
        model_config=None,
        trend_config=None,
    ):
        first_outcome = list(outcome_data.values())[0]
        super().__init__(
            X_media,
            first_outcome,
            channel_names,
            index,
            model_config=model_config,
            trend_config=trend_config,
        )

        self.outcome_data = outcome_data
        self.config = config
        self.mediator_data = mediator_data or {}
        self.mediator_masks = mediator_masks or {}
        self.promotion_data = promotion_data or {}

        self.mediator_names = [m.name for m in config.nested.mediators]
        self.outcome_names = [o.name for o in config.multivariate.outcomes]
        self.n_mediators = len(self.mediator_names)
        self.n_outcomes = len(self.outcome_names)

        # Per-outcome standardization (same convention as MultivariateMMM):
        # raw data attributes, standardized likelihood, original-unit
        # deterministics.
        self.outcome_means = {
            name: float(np.asarray(outcome_data[name], dtype=float).mean())
            for name in self.outcome_names
        }
        self.outcome_stds = {
            name: float(np.asarray(outcome_data[name], dtype=float).std()) + 1e-8
            for name in self.outcome_names
        }

        # Build cross-effect structure (same machinery as MultivariateMMM).
        self._cross_effect_specs = self._build_cross_effect_specs()

    def _build_coords(self) -> dict:
        coords = super()._build_coords()
        coords["mediator"] = self.mediator_names
        coords["outcome"] = self.outcome_names
        return coords

    def _build_cross_effect_specs(self):
        """Convert configured cross-effects to internal specs.

        Mirrors :meth:`MultivariateMMM._build_cross_effect_specs`: only the
        configured (source, target) directions get a free RV; everything else
        in the psi matrix is a structural zero (including the diagonal).
        """
        from ..components.cross_effects import CrossEffectSpec
        from ..config import CrossEffectType

        specs = []
        for ce in self.config.multivariate.cross_effects:
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

    def _baseline_dynamics_terms(self) -> tuple[dict, dict]:
        """Per-outcome trend + seasonality from the multivariate config's flags."""
        mv = self.config.multivariate
        return self._build_baseline_dynamics(
            mv.outcomes, mv.share_trend, mv.share_seasonality
        )

    def _build_model(self) -> pm.Model:
        """Build the combined model."""
        from ..components.cross_effects import (
            build_cross_effect_matrix,
            compute_cross_effect_contribution,
        )
        from ..components.priors import create_effect_prior
        from ..components.observation import (
            build_partial_observation_model,
            build_multivariate_likelihood,
        )

        coords = self._build_coords()

        with pm.Model(coords=coords) as model:
            # Data
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))

            Y_matrix = np.column_stack(
                [self.outcome_data[name] for name in self.outcome_names]
            )
            # ``Y`` stays raw (cross-effects act on the raw source outcome);
            # the likelihood fits standardized outcomes.
            means = np.array([self.outcome_means[n] for n in self.outcome_names])
            stds = np.array([self.outcome_stds[n] for n in self.outcome_names])
            Y_standardized = (Y_matrix - means) / stds
            Y = pm.Data("Y", Y_matrix, dims=("obs", "outcome"))

            # Media transformations: normalize -> geometric adstock -> logistic
            # saturation (the previously-unused ``alpha`` is now the carryover).
            media_transformed = []
            channel_tx = {}  # channel -> (x_sat, apply, x_input) for experiments
            for i, channel in enumerate(self.channel_names):
                x = X_media[:, i]
                alpha = pm.Beta(f"alpha_{channel}", alpha=2, beta=2)
                lam = pm.Gamma(f"lambda_{channel}", alpha=3, beta=1)
                apply = self._media_transform_apply(i, alpha, lam)
                x_sat = apply(x)
                media_transformed.append(x_sat)
                channel_tx[channel] = (x_sat, apply, x)

            media_transformed = pt.stack(media_transformed, axis=1)

            # Mediator models
            mediator_values = {}
            # Per-mediator media->mediator coefficient RVs (by reference).
            channel_mediator_betas: dict[str, dict] = {}
            for med_config in self.config.nested.mediators:
                med_name = med_config.name
                affecting = self._get_affecting_channels(med_name)

                alpha_med = pm.Normal(f"alpha_{med_name}", mu=0, sigma=2)

                # Media effects on mediator
                med_effect = pt.zeros(self.n_obs)
                med_betas: dict = {}
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
                        med_betas[ch] = beta
                channel_mediator_betas[med_name] = med_betas

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

            # Outcome parameters. Per-outcome intercept/media prior scales come
            # from each OutcomeConfig (defaults 2.0 / 0.5 reproduce the historical
            # shared-scalar priors); gamma's per-mediator scale comes from each
            # mediator's outcome_effect (matching NestedMMM — previously a fixed
            # 0.5). A spec/DAG prior override now reaches these RVs.
            outcomes = self.config.multivariate.outcomes
            mediators = self.config.nested.mediators
            intercept_sigma = np.array(
                [oc.intercept_prior_sigma for oc in outcomes], dtype=float
            )
            media_sigma = np.array(
                [oc.media_effect.sigma for oc in outcomes], dtype=float
            )
            alpha_y = pm.Normal("alpha_y", mu=0, sigma=intercept_sigma, dims="outcome")
            beta_direct = pm.Normal(
                "beta_direct",
                mu=0,
                sigma=media_sigma[:, None],
                dims=("outcome", "channel"),
            )

            # Mediator → Outcome effects
            gamma_sigma = np.array(
                [m.outcome_effect.sigma for m in mediators], dtype=float
            )
            gamma = pm.Normal(
                "gamma", mu=0, sigma=gamma_sigma[None, :], dims=("outcome", "mediator")
            )

            # Baseline dynamics (trend + seasonality) per outcome, or None.
            trend_terms, seasonality_terms = self._baseline_dynamics_terms()

            # Cross-effects: only configured (source, target) directions get a
            # free RV (sign-constrained where requested); all other entries —
            # including the diagonal — are structural zeros. Same machinery as
            # MultivariateMMM. No configured cross-effects => no psi RVs.
            if self._cross_effect_specs:
                psi_matrix, _psi_params = build_cross_effect_matrix(
                    self._cross_effect_specs,
                    self.n_outcomes,
                    name_prefix="psi",
                )
                pm.Deterministic("psi_matrix", psi_matrix)
            else:
                psi_matrix = None

            # Optional promotion modulation per source outcome (built once).
            modulation: dict[int, pt.TensorVariable] = {}
            if psi_matrix is not None:
                for ce in self.config.multivariate.cross_effects:
                    if (
                        ce.promotion_modulated
                        and ce.promotion_column
                        and ce.promotion_column in self.promotion_data
                    ):
                        source_idx = self.outcome_names.index(ce.source_outcome)
                        modulation[source_idx] = pm.Data(
                            f"promo_{ce.source_outcome}",
                            self.promotion_data[ce.promotion_column],
                            dims="obs",
                        )

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

                # Baseline dynamics (standardized scale)
                if trend_terms.get(k) is not None:
                    mu_k = mu_k + trend_terms[k]
                if seasonality_terms.get(k) is not None:
                    mu_k = mu_k + seasonality_terms[k]

                # Cross-effects: psi_matrix[source, target] * Y[:, source],
                # added to target k (same orientation as MultivariateMMM). The
                # raw-scale contribution is divided by the target's std so psi
                # keeps its raw source->target interpretation in the
                # standardized-mu graph.
                if psi_matrix is not None:
                    mu_k = mu_k + (
                        compute_cross_effect_contribution(
                            Y, psi_matrix, k, self.n_outcomes, modulation
                        )
                        / stds[k]
                    )

                mu_list.append(mu_k)

            mu_standardized = pt.stack(mu_list, axis=1)
            # Original-unit predictions (reports and oracles consume this)
            pm.Deterministic(
                "mu",
                mu_standardized * stds[None, :] + means[None, :],
                dims=("obs", "outcome"),
            )

            # Multivariate likelihood (standardized scale)
            build_multivariate_likelihood(
                "Y_obs",
                mu_standardized,
                Y_standardized,
                self.n_outcomes,
                self.config.multivariate.lkj_eta,
                dims=("obs", "outcome"),
            )

            # Derived quantities. ``indirect`` respects the mediator->outcome
            # routing (``_get_affected_outcomes``) so it matches ``mu`` exactly.
            # Registered in original units (x outcome std), matching ``mu``.
            for i, channel in enumerate(self.channel_names):
                for k, outcome_name in enumerate(self.outcome_names):
                    direct = beta_direct[k, i] * stds[k]

                    indirect = pt.zeros(())
                    for m, med_name in enumerate(self.mediator_names):
                        if outcome_name not in self._get_affected_outcomes(med_name):
                            continue
                        beta = channel_mediator_betas.get(med_name, {}).get(channel)
                        if beta is not None:
                            indirect = indirect + beta * gamma[k, m] * stds[k]

                    pm.Deterministic(f"direct_{channel}_{outcome_name}", direct)
                    pm.Deterministic(f"indirect_{channel}_{outcome_name}", indirect)
                    pm.Deterministic(
                        f"total_{channel}_{outcome_name}", direct + indirect
                    )

            # Experiment calibration: per (channel, outcome) the total effect is
            # the direct media coefficient plus the routed mediated paths --
            # coef = beta_direct[k,c] + sum_{m affecting k} beta_{c->m} * gamma[k,m]
            # -- applied to the channel's saturated spend (raw outcome scale).
            if self.experiments:
                handles: dict = {}
                for k, outcome_name in enumerate(self.outcome_names):
                    for c, channel in enumerate(self.channel_names):
                        coef = beta_direct[k, c]
                        for m, med_name in enumerate(self.mediator_names):
                            if outcome_name not in self._get_affected_outcomes(
                                med_name
                            ):
                                continue
                            beta = channel_mediator_betas.get(med_name, {}).get(channel)
                            if beta is not None:
                                coef = coef + beta * gamma[k, m]
                        x_sat, apply, x_input = channel_tx[channel]
                        handles[(channel, outcome_name)] = {
                            "coef": coef,
                            "x_sat": x_sat,
                            "apply": apply,
                            "x_input": x_input,
                            "spend_obs": self.X_media[:, c],
                            "scale": self.outcome_stds[outcome_name],
                        }
                self._add_experiment_likelihoods(handles)

        return model

    def get_cross_effects_summary(self) -> pd.DataFrame:
        """Extract cross-effect estimates.

        Same convention as :meth:`MultivariateMMM.get_cross_effects_summary`:
        each row reports the configured source -> target effect read from
        ``psi_matrix[source_idx, target_idx]`` (cannibalization rows carry the
        imposed negative sign).
        """

        self._check_fitted()

        if "psi_matrix" not in self._trace.posterior:
            return pd.DataFrame()

        results = []
        psi = self._trace.posterior["psi_matrix"]

        for spec in self._cross_effect_specs:
            vals = psi[:, :, spec.source_idx, spec.target_idx].values.flatten()
            from mmm_framework.utils.arviz_compat import hdi_bounds

            hdi = hdi_bounds(vals, 0.94)

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
