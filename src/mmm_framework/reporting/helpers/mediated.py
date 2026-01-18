"""
Mediated effects and cross-effects computation for extended MMM models.

Functions for computing direct, indirect, and total effects for nested
and multivariate models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .results import MediatedEffectResult
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
)


def compute_mediated_effects(
    model: Any,
    hdi_prob: float = 0.94,
) -> list[MediatedEffectResult]:
    """
    Compute direct, indirect, and total effects for nested/combined models.

    For models with mediators (e.g., Media -> Awareness -> Sales), this
    decomposes effects into direct and indirect pathways.

    Parameters
    ----------
    model : NestedMMM or CombinedMMM
        Fitted nested or combined model
    hdi_prob : float
        HDI probability

    Returns
    -------
    list[MediatedEffectResult]
        Effect decomposition by channel and outcome

    Examples
    --------
    >>> effects = compute_mediated_effects(nested_mmm)
    >>> for e in effects:
    ...     print(f"{e.channel}: {e.proportion_mediated:.0%} mediated")
    """
    _check_model_fitted(model)

    # Check if model has mediation
    if not hasattr(model, "get_mediation_effects") and not hasattr(
        model, "mediator_names"
    ):
        raise ValueError("Model does not support mediation analysis")

    # Try model's built-in method (but verify it returns valid data)
    if hasattr(model, "get_mediation_effects"):
        try:
            df = model.get_mediation_effects()
            # Verify it's actually a DataFrame with expected columns
            if (
                isinstance(df, pd.DataFrame)
                and len(df) > 0
                and "channel" in df.columns
            ):
                return _convert_mediation_df(df, hdi_prob)
        except Exception as e:
            logger.debug(
                f"Model mediation method failed or returned invalid data: {e}"
            )

    # Manual computation from trace
    return _compute_mediation_from_trace(model, hdi_prob)


def _convert_mediation_df(
    df: pd.DataFrame, hdi_prob: float
) -> list[MediatedEffectResult]:
    """Convert model's mediation DataFrame to MediatedEffectResult list."""
    results = []

    for _, row in df.iterrows():
        total = row.get("total_effect", 0)
        direct = row.get("direct_effect", 0)
        indirect = row.get("total_indirect", total - direct)

        prop_mediated = indirect / total if total != 0 else 0.0

        results.append(
            MediatedEffectResult(
                channel=row.get("channel", ""),
                outcome=row.get("outcome", "sales"),
                direct_mean=direct,
                direct_lower=direct - 1.96 * row.get("direct_effect_sd", 0),
                direct_upper=direct + 1.96 * row.get("direct_effect_sd", 0),
                indirect_mean=indirect,
                indirect_lower=indirect,  # Would need samples for proper HDI
                indirect_upper=indirect,
                total_mean=total,
                total_lower=total,
                total_upper=total,
                proportion_mediated=prop_mediated,
                mediator_breakdown=row.get("indirect_effects", {}),
            )
        )

    return results


def _compute_mediation_from_trace(
    model: Any,
    hdi_prob: float,
) -> list[MediatedEffectResult]:
    """Compute mediation effects from trace."""
    posterior = _get_posterior(model)
    channels = _get_channel_names(model)

    if posterior is None:
        logger.warning("No posterior found for mediation computation")
        return []

    results = []

    # Get outcome names
    outcome_names = getattr(model, "outcome_names", ["sales"])
    if isinstance(outcome_names, str):
        outcome_names = [outcome_names]

    # Helper to check if variable exists in posterior
    def _var_exists(var_name: str) -> bool:
        try:
            if hasattr(posterior, "data_vars"):
                return var_name in posterior.data_vars
            return var_name in posterior
        except Exception:
            return False

    # Helper to get samples from posterior
    def _get_samples(var_name: str) -> np.ndarray | None:
        try:
            if _var_exists(var_name):
                return _flatten_samples(posterior[var_name].values)
        except Exception as e:
            logger.debug(f"Could not get samples for {var_name}: {e}")
        return None

    for channel in channels:
        for outcome in outcome_names:
            # Look for direct/indirect/total deterministics
            direct_var = f"direct_{channel}_{outcome}"
            indirect_var = f"indirect_{channel}_{outcome}"
            total_var = f"total_{channel}_{outcome}"

            direct_samples = _get_samples(direct_var)
            total_samples = _get_samples(total_var)

            if direct_samples is not None and total_samples is not None:
                indirect_samples = _get_samples(indirect_var)
                if indirect_samples is None:
                    indirect_samples = total_samples - direct_samples

                direct_mean = float(np.mean(direct_samples))
                direct_lower, direct_upper = _compute_hdi(direct_samples, hdi_prob)

                indirect_mean = float(np.mean(indirect_samples))
                indirect_lower, indirect_upper = _compute_hdi(
                    indirect_samples, hdi_prob
                )

                total_mean = float(np.mean(total_samples))
                total_lower, total_upper = _compute_hdi(total_samples, hdi_prob)

                prop_mediated = (
                    indirect_mean / total_mean if total_mean != 0 else 0.0
                )

                results.append(
                    MediatedEffectResult(
                        channel=channel,
                        outcome=outcome,
                        direct_mean=direct_mean,
                        direct_lower=direct_lower,
                        direct_upper=direct_upper,
                        indirect_mean=indirect_mean,
                        indirect_lower=indirect_lower,
                        indirect_upper=indirect_upper,
                        total_mean=total_mean,
                        total_lower=total_lower,
                        total_upper=total_upper,
                        proportion_mediated=prop_mediated,
                    )
                )

    return results


def compute_cross_effects(
    model: Any,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """
    Compute cross-effects between outcomes for multivariate models.

    For models with multiple outcomes (e.g., Product A sales, Product B sales),
    this extracts the cross-effect coefficients showing how one outcome
    influences another.

    Parameters
    ----------
    model : MultivariateMMM or CombinedMMM
        Fitted multivariate model
    hdi_prob : float
        HDI probability

    Returns
    -------
    pd.DataFrame
        Cross-effect matrix with uncertainty
    """
    _check_model_fitted(model)

    if not hasattr(model, "outcome_names"):
        raise ValueError("Model does not support cross-effect analysis")

    # Try model's built-in method
    if hasattr(model, "get_cross_effect_summary"):
        try:
            result = model.get_cross_effect_summary()
            if result is not None and len(result) > 0:
                return result
        except Exception as e:
            logger.warning(f"Model cross-effect method failed: {e}")

    # Manual computation from trace
    posterior = _get_posterior(model)
    outcome_names = model.outcome_names

    if posterior is None:
        logger.warning("No posterior found for cross-effect computation")
        return pd.DataFrame()

    rows = []

    # Helper to check if variable exists in posterior
    def _var_exists(var_name: str) -> bool:
        try:
            if hasattr(posterior, "data_vars"):
                return var_name in posterior.data_vars
            return var_name in posterior
        except Exception:
            return False

    # Look for psi matrix or psi_matrix (different naming conventions)
    psi_var_name = None
    for name in ["psi", "psi_matrix", "cross_effects"]:
        if _var_exists(name):
            psi_var_name = name
            break

    if psi_var_name is not None:
        try:
            psi_samples = posterior[psi_var_name].values

            # Handle different shapes - could be (chain, draw, outcome, outcome)
            # or just (samples, outcome, outcome)
            if psi_samples.ndim == 4:
                # (chain, draw, outcome, outcome)
                for i, source in enumerate(outcome_names):
                    for j, target in enumerate(outcome_names):
                        if i != j:
                            samples = psi_samples[:, :, i, j].flatten()
                            mean = float(np.mean(samples))
                            lower, upper = _compute_hdi(samples, hdi_prob)

                            rows.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "effect_mean": mean,
                                    "effect_hdi_low": lower,
                                    "effect_hdi_high": upper,
                                    "prob_positive": float(np.mean(samples > 0)),
                                }
                            )
            elif psi_samples.ndim == 3:
                # (samples, outcome, outcome)
                for i, source in enumerate(outcome_names):
                    for j, target in enumerate(outcome_names):
                        if i != j:
                            samples = psi_samples[:, i, j].flatten()
                            mean = float(np.mean(samples))
                            lower, upper = _compute_hdi(samples, hdi_prob)

                            rows.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "effect_mean": mean,
                                    "effect_hdi_low": lower,
                                    "effect_hdi_high": upper,
                                    "prob_positive": float(np.mean(samples > 0)),
                                }
                            )
        except Exception as e:
            logger.warning(f"Error extracting cross-effects from {psi_var_name}: {e}")

    return pd.DataFrame(rows)


__all__ = [
    "compute_mediated_effects",
    "compute_cross_effects",
    "_convert_mediation_df",
    "_compute_mediation_from_trace",
]
