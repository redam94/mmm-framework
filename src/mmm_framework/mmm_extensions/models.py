"""
Nested and Multivariate Model Implementations

Uses modular components from components.py and configurations from config.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from dataclasses import dataclass, field
from typing import Any

from .config import (
    MediatorType,
    CrossEffectType,
    EffectConstraint,
    SaturationType,
    MediatorConfig,
    OutcomeConfig,
    CrossEffectConfig,
    NestedModelConfig,
    MultivariateModelConfig,
    CombinedModelConfig,
)
from .components import (
    # Transformations
    geometric_adstock,
    logistic_saturation,
    hill_saturation,
    # Priors
    create_adstock_prior,
    create_saturation_prior,
    create_effect_prior,
    # Builders
    build_media_transforms,
    build_linear_effect,
    build_gaussian_likelihood,
    build_partial_observation_model,
    build_multivariate_likelihood,
    build_cross_effect_matrix,
    compute_cross_effect_contribution,
    # Results
    MediaTransformResult,
    EffectResult,
    CrossEffectSpec,
)


# =============================================================================
# Result Containers
# =============================================================================

@dataclass
class MediationEffects:
    """Container for mediation analysis results."""
    channel: str
    direct_effect: float
    direct_effect_sd: float
    indirect_effects: dict[str, float]  # mediator -> effect
    total_indirect: float
    total_effect: float
    proportion_mediated: float
    
    def to_dict(self) -> dict:
        result = {
            "channel": self.channel,
            "direct_effect": self.direct_effect,
            "direct_effect_sd": self.direct_effect_sd,
            "total_indirect": self.total_indirect,
            "total_effect": self.total_effect,
            "proportion_mediated": self.proportion_mediated,
        }
        for med, eff in self.indirect_effects.items():
            result[f"indirect_via_{med}"] = eff
        return result


@dataclass
class CrossEffectSummary:
    """Container for cross-effect analysis results."""
    source: str
    target: str
    effect_type: str
    mean: float
    sd: float
    hdi_low: float
    hdi_high: float


@dataclass
class ModelResults:
    """Container for fitted model results."""
    trace: az.InferenceData
    model: pm.Model
    config: Any
    
    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary statistics."""
        return az.summary(self.trace, var_names=var_names)
    
    def plot_trace(self, var_names: list[str] | None = None, **kwargs):
        """Plot trace diagnostics."""
        return az.plot_trace(self.trace, var_names=var_names, **kwargs)


# =============================================================================
# Base Model Class
# =============================================================================

class BaseExtendedMMM:
    """Base class for extended MMM models."""
    
    def __init__(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        index: pd.Index | None = None,
    ):
        self.X_media = X_media
        self.y = y
        self.channel_names = channel_names
        self.index = index if index is not None else pd.RangeIndex(len(y))
        
        self.n_obs = len(y)
        self.n_channels = len(channel_names)
        
        self._model: pm.Model | None = None
        self._trace: az.InferenceData | None = None
    
    def _build_coords(self) -> dict:
        """Build PyMC coordinates. Override in subclasses."""
        return {
            "obs": np.arange(self.n_obs),
            "channel": self.channel_names,
        }
    
    def _build_model(self) -> pm.Model:
        """Build the PyMC model. Override in subclasses."""
        raise NotImplementedError
    
    @property
    def model(self) -> pm.Model:
        """Get or build the PyMC model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model
    
    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        **kwargs,
    ) -> ModelResults:
        """Fit the model using MCMC."""
        with self.model:
            self._trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                **kwargs,
            )
        return ModelResults(
            trace=self._trace,
            model=self.model,
            config=getattr(self, 'config', None),
        )
    
    def _check_fitted(self):
        """Check that model has been fitted."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")


# =============================================================================
# Nested/Mediated Model
# =============================================================================

class NestedMMM(BaseExtendedMMM):
    """
    MMM with nested/mediated causal pathways.
    
    Models: Media → Mediator(s) → Outcome
    With optional direct effects: Media → Outcome
    """
    
    def __init__(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        config: NestedModelConfig,
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
        med_config: MediatorConfig,
        media_transformed: pt.TensorVariable,
    ) -> pt.TensorVariable:
        """Build model for a single mediator."""
        med_name = med_config.name
        
        # Get affecting channels
        affecting = self._get_affecting_channels(med_name)
        channel_indices = [
            self.channel_names.index(c) 
            for c in affecting 
            if c in self.channel_names
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
        if med_config.mediator_type != MediatorType.FULLY_LATENT:
            if med_name in self.mediator_data:
                obs_data = self.mediator_data[med_name]
                mask = self.mediator_masks.get(
                    med_name,
                    ~np.isnan(obs_data) if obs_data is not None else np.zeros(self.n_obs, dtype=bool)
                )
                
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
        coords = self._build_coords()
        
        with pm.Model(coords=coords) as model:
            # Data
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))
            y = pm.Data("y", self.y, dims="obs")
            
            # Media transformations (simplified - full version would use components)
            media_transformed = []
            for i, channel in enumerate(self.channel_names):
                x = X_media[:, i]
                
                alpha = pm.Beta(f"alpha_{channel}", alpha=2, beta=2)
                lam = pm.Gamma(f"lambda_{channel}", alpha=3, beta=1)
                
                x_sat = logistic_saturation(x, lam)  # Simplified: no adstock scan
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
                    m.allow_direct_effect and 
                    channel in self._get_affecting_channels(m.name)
                    for m in self.config.mediators
                )
                
                if has_direct:
                    delta = pm.Normal(f"delta_direct_{channel}", mu=0, sigma=0.5)
                    direct_contrib = direct_contrib + delta * media_transformed[:, i]
                    pm.Deterministic(
                        f"direct_effect_{channel}",
                        delta * media_transformed[:, i],
                        dims="obs"
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
            
            results.append(MediationEffects(
                channel=channel,
                direct_effect=direct,
                direct_effect_sd=direct_sd,
                indirect_effects=indirect_effects,
                total_indirect=total_indirect,
                total_effect=total,
                proportion_mediated=prop_mediated,
            ))
        
        return pd.DataFrame([r.to_dict() for r in results])


# =============================================================================
# Multivariate Outcome Model
# =============================================================================

class MultivariateMMM(BaseExtendedMMM):
    """
    MMM with multiple correlated outcomes and cross-effects.
    """
    
    def __init__(
        self,
        X_media: np.ndarray,
        outcome_data: dict[str, np.ndarray],
        channel_names: list[str],
        config: MultivariateModelConfig,
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
    
    def _build_cross_effect_specs(self) -> list[CrossEffectSpec]:
        """Convert config cross-effects to internal specs."""
        specs = []
        for ce in self.config.cross_effects:
            source_idx = self.outcome_names.index(ce.source_outcome)
            target_idx = self.outcome_names.index(ce.target_outcome)
            
            specs.append(CrossEffectSpec(
                source_idx=source_idx,
                target_idx=target_idx,
                effect_type=ce.effect_type.value,
                prior_sigma=ce.prior_sigma,
            ))
            
            # Add reverse direction for symmetric effects
            if ce.effect_type == CrossEffectType.SYMMETRIC:
                specs.append(CrossEffectSpec(
                    source_idx=target_idx,
                    target_idx=source_idx,
                    effect_type=ce.effect_type.value,
                    prior_sigma=ce.prior_sigma,
                ))
        
        return specs
    
    def _build_model(self) -> pm.Model:
        """Build the multivariate model."""
        coords = self._build_coords()
        
        with pm.Model(coords=coords) as model:
            # Data
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))
            
            Y_matrix = np.column_stack([
                self.outcome_data[name] for name in self.outcome_names
            ])
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
                "beta_media", mu=0, sigma=0.5, 
                dims=("outcome", "channel")
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
                                    dims="obs"
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
        self._check_fitted()
        
        if "psi_matrix" not in self._trace.posterior:
            return pd.DataFrame()
        
        results = []
        psi = self._trace.posterior["psi_matrix"]
        
        for spec in self._cross_effect_specs:
            vals = psi[:, :, spec.source_idx, spec.target_idx].values.flatten()
            hdi = az.hdi(vals, hdi_prob=0.94)
            
            results.append(CrossEffectSummary(
                source=self.outcome_names[spec.source_idx],
                target=self.outcome_names[spec.target_idx],
                effect_type=spec.effect_type,
                mean=float(np.mean(vals)),
                sd=float(np.std(vals)),
                hdi_low=float(hdi[0]),
                hdi_high=float(hdi[1]),
            ))
        
        return pd.DataFrame([
            {
                "source": r.source,
                "target": r.target,
                "effect_type": r.effect_type,
                "mean": r.mean,
                "sd": r.sd,
                "hdi_3%": r.hdi_low,
                "hdi_97%": r.hdi_high,
            }
            for r in results
        ])
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Extract posterior mean correlation matrix."""
        self._check_fitted()
        
        corr = self._trace.posterior["Y_obs_correlation"].mean(
            dim=["chain", "draw"]
        ).values
        
        return pd.DataFrame(
            corr,
            index=self.outcome_names,
            columns=self.outcome_names,
        )


# =============================================================================
# Combined Model
# =============================================================================

class CombinedMMM(BaseExtendedMMM):
    """
    Combined nested + multivariate model.
    
    Supports both mediated pathways AND correlated outcomes with cross-effects.
    """
    
    def __init__(
        self,
        X_media: np.ndarray,
        outcome_data: dict[str, np.ndarray],
        channel_names: list[str],
        config: CombinedModelConfig,
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
        coords = self._build_coords()
        
        with pm.Model(coords=coords) as model:
            # Data
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))
            
            Y_matrix = np.column_stack([
                self.outcome_data[name] for name in self.outcome_names
            ])
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
                    mask = self.mediator_masks.get(
                        med_name,
                        ~np.isnan(obs_data)
                    )
                    build_partial_observation_model(
                        med_name, mediator_latent, obs_data, mask,
                        med_config.observation_noise_sigma
                    )
                
                mediator_values[med_name] = mediator_latent
                pm.Deterministic(f"{med_name}_latent", mediator_latent, dims="obs")
            
            # Outcome parameters
            alpha_y = pm.Normal("alpha_y", mu=0, sigma=2, dims="outcome")
            beta_direct = pm.Normal(
                "beta_direct", mu=0, sigma=0.5,
                dims=("outcome", "channel")
            )
            
            # Mediator → Outcome effects
            gamma = pm.Normal(
                "gamma", mu=0, sigma=0.5,
                dims=("outcome", "mediator")
            )
            
            # Cross-effects
            psi = pm.Normal(
                "psi", mu=0, sigma=0.3,
                dims=("outcome", "outcome")
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
                
                # Cross-effects
                for j in range(self.n_outcomes):
                    if j != k:
                        mu_k = mu_k + psi[j, k] * Y[:, j]
                
                mu_list.append(mu_k)
            
            mu = pt.stack(mu_list, axis=1)
            
            # Multivariate likelihood
            build_multivariate_likelihood(
                "Y_obs", mu, Y_matrix, self.n_outcomes,
                self.config.multivariate.lkj_eta,
                dims=("obs", "outcome")
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
                        f"total_{channel}_{outcome_name}",
                        direct + indirect
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