"""
Variable selection priors for MMM Extensions.

Provides regularized horseshoe, spike-and-slab, and Bayesian LASSO
priors for control variable selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd

    from ..config import (
        HorseshoeConfig,
        SpikeSlabConfig,
        LassoConfig,
        VariableSelectionConfig,
        VariableSelectionMethod,
    )


@dataclass
class VariableSelectionResult:
    """
    Container for variable selection prior outputs.

    Attributes
    ----------
    beta : pt.TensorVariable
        The coefficient vector with shrinkage/selection applied.
    inclusion_indicators : pt.TensorVariable | None
        For spike-slab: soft inclusion indicators (gamma).
    local_shrinkage : pt.TensorVariable | None
        For horseshoe: local shrinkage parameters (lambda).
    global_shrinkage : pt.TensorVariable | None
        For horseshoe: global shrinkage parameter (tau).
    effective_nonzero : pt.TensorVariable | None
        Estimated number of effectively nonzero coefficients.
    kappa : pt.TensorVariable | None
        Shrinkage factors for each coefficient (horseshoe).
    """

    beta: pt.TensorVariable
    inclusion_indicators: pt.TensorVariable | None = None
    local_shrinkage: pt.TensorVariable | None = None
    global_shrinkage: pt.TensorVariable | None = None
    effective_nonzero: pt.TensorVariable | None = None
    kappa: pt.TensorVariable | None = None


@dataclass
class ControlEffectResult:
    """
    Container for control variable effects with optional selection.

    Attributes
    ----------
    contribution : pt.TensorVariable
        Total control contribution (n_obs,).
    beta_selected : pt.TensorVariable | None
        Coefficients for selected (shrinkage) variables.
    beta_fixed : pt.TensorVariable | None
        Coefficients for fixed (non-shrinkage) variables.
    selection_result : VariableSelectionResult | None
        Full selection result for diagnostics.
    components : dict[str, pt.TensorVariable]
        Individual variable contributions.
    """

    contribution: pt.TensorVariable
    beta_selected: pt.TensorVariable | None = None
    beta_fixed: pt.TensorVariable | None = None
    selection_result: VariableSelectionResult | None = None
    components: dict[str, pt.TensorVariable] = field(default_factory=dict)


def create_regularized_horseshoe_prior(
    name: str,
    n_variables: int,
    n_obs: int,
    sigma: pt.TensorVariable,
    config: "HorseshoeConfig",
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create regularized horseshoe prior (Piironen & Vehtari, 2017).

    The regularized horseshoe provides:
    - Strong shrinkage of small effects toward zero
    - Minimal shrinkage of large effects (they "escape" the horseshoe)
    - Slab regularization to prevent unrealistically large effects

    Parameters
    ----------
    name : str
        Base name for the parameters.
    n_variables : int
        Number of variables (D).
    n_obs : int
        Number of observations (N).
    sigma : pt.TensorVariable
        Observation noise standard deviation.
    config : HorseshoeConfig
        Horseshoe configuration.
    dims : str | None
        PyMC dimension name for coefficients.

    Returns
    -------
    VariableSelectionResult
        Container with beta and diagnostic quantities.
    """
    D = n_variables
    D0 = min(config.expected_nonzero, D - 1)
    N = n_obs

    # Global shrinkage scale (Piironen & Vehtari recommendation)
    tau0 = (D0 / (D - D0)) * (sigma / np.sqrt(N))

    # Global shrinkage parameter
    tau = pm.HalfStudentT(
        f"{name}_tau",
        nu=config.global_df,
        sigma=tau0,
    )

    # Local shrinkage parameters
    dim_kwargs = {"dims": dims} if dims else {"shape": D}
    lambda_local = pm.HalfStudentT(
        f"{name}_lambda",
        nu=config.local_df,
        **dim_kwargs,
    )

    # Slab regularization (c^2)
    c2 = pm.InverseGamma(
        f"{name}_c2",
        alpha=config.slab_df / 2,
        beta=config.slab_df * config.slab_scale**2 / 2,
    )

    # Regularized local shrinkage
    lambda_tilde = pt.sqrt(c2) * lambda_local / pt.sqrt(c2 + tau**2 * lambda_local**2)

    # Standardized coefficients (non-centered parameterization)
    z = pm.Normal(f"{name}_z", mu=0, sigma=1, **dim_kwargs)

    # Final coefficients
    beta = pm.Deterministic(
        f"{name}",
        z * tau * lambda_tilde,
        dims=dims,
    )

    # Shrinkage factors kappa_j = 1 / (1 + tau^2 * lambda_j^2)
    kappa = pm.Deterministic(
        f"{name}_kappa",
        1 / (1 + tau**2 * lambda_local**2),
        dims=dims,
    )

    # Effective number of nonzero coefficients
    effective_nonzero = pm.Deterministic(
        f"{name}_effective_nonzero",
        pt.sum(1 - kappa),
    )

    return VariableSelectionResult(
        beta=beta,
        local_shrinkage=lambda_local,
        global_shrinkage=tau,
        effective_nonzero=effective_nonzero,
        kappa=kappa,
    )


def create_finnish_horseshoe_prior(
    name: str,
    n_variables: int,
    n_obs: int,
    sigma: pt.TensorVariable,
    config: "HorseshoeConfig",
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create Finnish horseshoe prior (Piironen & Vehtari, 2017).

    Mathematically identical to the regularized horseshoe.
    """
    return create_regularized_horseshoe_prior(
        name=name,
        n_variables=n_variables,
        n_obs=n_obs,
        sigma=sigma,
        config=config,
        dims=dims,
    )


def create_spike_slab_prior(
    name: str,
    n_variables: int,
    config: "SpikeSlabConfig",
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create spike-and-slab prior for variable selection.

    Parameters
    ----------
    name : str
        Base name for the parameters.
    n_variables : int
        Number of variables.
    config : SpikeSlabConfig
        Spike-slab configuration.
    dims : str | None
        PyMC dimension name.

    Returns
    -------
    VariableSelectionResult
        Container with beta and inclusion indicators.
    """
    dim_kwargs = {"dims": dims} if dims else {"shape": n_variables}

    if config.use_continuous_relaxation:
        # Continuous relaxation for NUTS
        prior_logit = np.log(
            config.prior_inclusion_prob / (1 - config.prior_inclusion_prob)
        )

        logit_gamma = pm.Normal(
            f"{name}_logit_gamma",
            mu=prior_logit,
            sigma=1.0,
            **dim_kwargs,
        )

        gamma = pm.Deterministic(
            f"{name}_gamma",
            pm.math.sigmoid(logit_gamma / config.temperature),
            dims=dims,
        )

        beta_slab = pm.Normal(
            f"{name}_slab",
            mu=0,
            sigma=config.slab_scale,
            **dim_kwargs,
        )

        beta_spike = pm.Normal(
            f"{name}_spike",
            mu=0,
            sigma=config.spike_scale,
            **dim_kwargs,
        )

        beta = pm.Deterministic(
            f"{name}",
            gamma * beta_slab + (1 - gamma) * beta_spike,
            dims=dims,
        )

    else:
        # Discrete spike-and-slab
        gamma = pm.Bernoulli(
            f"{name}_gamma",
            p=config.prior_inclusion_prob,
            **dim_kwargs,
        )

        beta_slab = pm.Normal(
            f"{name}_slab",
            mu=0,
            sigma=config.slab_scale,
            **dim_kwargs,
        )

        beta = pm.Deterministic(
            f"{name}",
            gamma * beta_slab,
            dims=dims,
        )

    effective_nonzero = pm.Deterministic(
        f"{name}_effective_nonzero",
        pt.sum(gamma),
    )

    return VariableSelectionResult(
        beta=beta,
        inclusion_indicators=gamma,
        effective_nonzero=effective_nonzero,
    )


def create_bayesian_lasso_prior(
    name: str,
    n_variables: int,
    config: "LassoConfig",
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Create Bayesian LASSO prior (Park & Casella, 2008).

    Parameters
    ----------
    name : str
        Base name for the parameters.
    n_variables : int
        Number of variables.
    config : LassoConfig
        LASSO configuration.
    dims : str | None
        PyMC dimension name.

    Returns
    -------
    VariableSelectionResult
        Container with beta and scale parameters.
    """
    dim_kwargs = {"dims": dims} if dims else {"shape": n_variables}

    # Scale mixture representation of Laplace
    tau = pm.Exponential(
        f"{name}_tau",
        lam=config.regularization**2 / 2,
        **dim_kwargs,
    )

    beta = pm.Normal(
        f"{name}",
        mu=0,
        sigma=pt.sqrt(tau),
        **dim_kwargs,
    )

    return VariableSelectionResult(
        beta=beta,
        local_shrinkage=tau,
    )


def create_variable_selection_prior(
    name: str,
    n_variables: int,
    n_obs: int,
    sigma: pt.TensorVariable,
    config: "VariableSelectionConfig",
    dims: str | None = None,
) -> VariableSelectionResult:
    """
    Factory function to create variable selection priors.

    Parameters
    ----------
    name : str
        Base name for the coefficient parameters.
    n_variables : int
        Number of control variables subject to selection.
    n_obs : int
        Number of observations.
    sigma : pt.TensorVariable
        Observation noise standard deviation.
    config : VariableSelectionConfig
        Complete configuration specifying method and hyperparameters.
    dims : str | None
        PyMC dimension name for the coefficient vector.

    Returns
    -------
    VariableSelectionResult
        Container with coefficient vector and diagnostic quantities.
    """
    from ..config import VariableSelectionMethod

    method = config.method

    if method == VariableSelectionMethod.NONE:
        dim_kwargs = {"dims": dims} if dims else {"shape": n_variables}
        beta = pm.Normal(
            name,
            mu=0,
            sigma=0.5,
            **dim_kwargs,
        )
        return VariableSelectionResult(beta=beta)

    elif method == VariableSelectionMethod.REGULARIZED_HORSESHOE:
        return create_regularized_horseshoe_prior(
            name=name,
            n_variables=n_variables,
            n_obs=n_obs,
            sigma=sigma,
            config=config.horseshoe,
            dims=dims,
        )

    elif method == VariableSelectionMethod.FINNISH_HORSESHOE:
        return create_finnish_horseshoe_prior(
            name=name,
            n_variables=n_variables,
            n_obs=n_obs,
            sigma=sigma,
            config=config.horseshoe,
            dims=dims,
        )

    elif method == VariableSelectionMethod.SPIKE_SLAB:
        return create_spike_slab_prior(
            name=name,
            n_variables=n_variables,
            config=config.spike_slab,
            dims=dims,
        )

    elif method == VariableSelectionMethod.BAYESIAN_LASSO:
        return create_bayesian_lasso_prior(
            name=name,
            n_variables=n_variables,
            config=config.lasso,
            dims=dims,
        )

    else:
        raise ValueError(f"Unknown variable selection method: {method}")


def build_control_effects_with_selection(
    X_controls: np.ndarray | pt.TensorVariable,
    control_names: list[str],
    n_obs: int,
    sigma: pt.TensorVariable,
    selection_config: "VariableSelectionConfig",
    name_prefix: str = "control",
) -> ControlEffectResult:
    """
    Build control variable effects with optional variable selection.

    Parameters
    ----------
    X_controls : array-like
        Control variable matrix (n_obs, n_controls).
    control_names : list[str]
        Names of control variables.
    n_obs : int
        Number of observations.
    sigma : pt.TensorVariable
        Observation noise (for horseshoe calibration).
    selection_config : VariableSelectionConfig
        Configuration for variable selection.
    name_prefix : str
        Prefix for parameter names.

    Returns
    -------
    ControlEffectResult
        Container with contributions and coefficients.
    """
    from ..config import VariableSelectionMethod

    X_controls = pt.as_tensor_variable(X_controls)

    # Partition variables
    selectable, fixed = selection_config.get_selectable_variables(control_names)

    components = {}
    contribution_parts = []
    beta_selected = None
    beta_fixed = None
    selection_result = None

    # Handle fixed variables
    if fixed:
        fixed_idx = [control_names.index(v) for v in fixed]
        X_fixed = X_controls[:, fixed_idx]

        beta_fixed = pm.Normal(
            f"{name_prefix}_fixed",
            mu=0,
            sigma=0.5,
            dims=f"{name_prefix}_fixed_dim" if len(fixed) > 1 else None,
            shape=len(fixed) if len(fixed) > 1 else (),
        )

        if len(fixed) == 1:
            fixed_contrib = beta_fixed * X_fixed[:, 0]
            components[fixed[0]] = fixed_contrib
        else:
            fixed_contrib = pt.dot(X_fixed, beta_fixed)
            for i, var_name in enumerate(fixed):
                components[var_name] = (
                    beta_fixed[i] * X_controls[:, control_names.index(var_name)]
                )

        contribution_parts.append(fixed_contrib)

    # Handle selectable variables
    if selectable and selection_config.method != VariableSelectionMethod.NONE:
        selectable_idx = [control_names.index(v) for v in selectable]
        X_selectable = X_controls[:, selectable_idx]

        selection_result = create_variable_selection_prior(
            name=f"{name_prefix}_select",
            n_variables=len(selectable),
            n_obs=n_obs,
            sigma=sigma,
            config=selection_config,
            dims=f"{name_prefix}_select_dim",
        )
        beta_selected = selection_result.beta

        selectable_contrib = pt.dot(X_selectable, beta_selected)
        contribution_parts.append(selectable_contrib)

        for i, var_name in enumerate(selectable):
            components[var_name] = (
                beta_selected[i] * X_controls[:, control_names.index(var_name)]
            )

    elif selectable:
        selectable_idx = [control_names.index(v) for v in selectable]
        X_selectable = X_controls[:, selectable_idx]

        beta_selected = pm.Normal(
            f"{name_prefix}_select",
            mu=0,
            sigma=0.5,
            shape=len(selectable),
        )

        selectable_contrib = pt.dot(X_selectable, beta_selected)
        contribution_parts.append(selectable_contrib)

        for i, var_name in enumerate(selectable):
            components[var_name] = (
                beta_selected[i] * X_controls[:, control_names.index(var_name)]
            )

    # Combine contributions
    if contribution_parts:
        total_contribution = sum(contribution_parts)
    else:
        total_contribution = pt.zeros(n_obs)

    return ControlEffectResult(
        contribution=total_contribution,
        beta_selected=beta_selected,
        beta_fixed=beta_fixed,
        selection_result=selection_result,
        components=components,
    )


def compute_inclusion_probabilities(
    trace: "az.InferenceData",
    config: "VariableSelectionConfig",
    name: str = "beta_controls",
    threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Compute posterior inclusion probabilities from fitted model.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples from fitted model.
    config : VariableSelectionConfig
        Configuration used for fitting.
    name : str
        Base name of the coefficient parameters.
    threshold : float
        For horseshoe: signal-to-noise threshold for "inclusion".

    Returns
    -------
    dict
        Dictionary with 'inclusion_prob' array and 'effective_nonzero'.
    """
    from ..config import VariableSelectionMethod

    posterior = trace.posterior

    if config.method == VariableSelectionMethod.SPIKE_SLAB:
        gamma_name = f"{name}_gamma"
        if gamma_name in posterior:
            gamma_samples = posterior[gamma_name].values
            inclusion_prob = gamma_samples.mean(axis=(0, 1))
            effective_nonzero = inclusion_prob.sum()
        else:
            raise ValueError(f"Could not find {gamma_name} in posterior")

    elif config.method in [
        VariableSelectionMethod.REGULARIZED_HORSESHOE,
        VariableSelectionMethod.FINNISH_HORSESHOE,
    ]:
        kappa_name = f"{name}_kappa"
        if kappa_name in posterior:
            kappa_samples = posterior[kappa_name].values
            inclusion_prob = 1 - kappa_samples.mean(axis=(0, 1))
            effective_nonzero = inclusion_prob.sum()
        else:
            beta_samples = posterior[name].values
            beta_mean = np.abs(beta_samples.mean(axis=(0, 1)))
            beta_std = beta_samples.std(axis=(0, 1)) + 1e-10
            snr = beta_mean / beta_std
            inclusion_prob = (snr > threshold).astype(float)
            effective_nonzero = inclusion_prob.sum()

    else:
        beta_samples = posterior[name].values
        lower = np.percentile(beta_samples, 2.5, axis=(0, 1))
        upper = np.percentile(beta_samples, 97.5, axis=(0, 1))
        inclusion_prob = ((lower > 0) | (upper < 0)).astype(float)
        effective_nonzero = inclusion_prob.sum()

    return {
        "inclusion_prob": inclusion_prob,
        "effective_nonzero": effective_nonzero,
    }


def summarize_variable_selection(
    trace: "az.InferenceData",
    control_names: list[str],
    config: "VariableSelectionConfig",
    name: str = "beta_controls",
) -> "pd.DataFrame":
    """
    Create summary table of variable selection results.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples.
    control_names : list[str]
        Names of control variables.
    config : VariableSelectionConfig
        Configuration used.
    name : str
        Base parameter name.

    Returns
    -------
    pd.DataFrame
        Summary with columns: variable, mean, std, hdi_3%, hdi_97%,
        inclusion_prob, selected.
    """
    import pandas as pd

    posterior = trace.posterior
    beta_samples = posterior[name].values

    inclusion_info = compute_inclusion_probabilities(trace, config, name)

    summary_data = []
    for i, var_name in enumerate(control_names):
        var_samples = beta_samples[:, :, i].flatten()
        summary_data.append(
            {
                "variable": var_name,
                "mean": var_samples.mean(),
                "std": var_samples.std(),
                "hdi_3%": np.percentile(var_samples, 3),
                "hdi_97%": np.percentile(var_samples, 97),
                "inclusion_prob": inclusion_info["inclusion_prob"][i],
                "selected": inclusion_info["inclusion_prob"][i] > 0.5,
            }
        )

    df = pd.DataFrame(summary_data)
    df = df.sort_values("inclusion_prob", ascending=False)

    return df


__all__ = [
    "VariableSelectionResult",
    "ControlEffectResult",
    "create_regularized_horseshoe_prior",
    "create_finnish_horseshoe_prior",
    "create_spike_slab_prior",
    "create_bayesian_lasso_prior",
    "create_variable_selection_prior",
    "build_control_effects_with_selection",
    "compute_inclusion_probabilities",
    "summarize_variable_selection",
]
