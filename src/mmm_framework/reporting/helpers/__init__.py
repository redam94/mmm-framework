"""
Reporting Helper Functions for MMM Framework.

This module provides comprehensive helper functions for computing and visualizing
key MMM outputs with proper uncertainty quantification:

- ROI computation with credible intervals
- Prior vs posterior comparison plots
- Adstock and saturation effect visualization
- Component decomposition analysis
- Extended model support (NestedMMM, MultivariateMMM, CombinedMMM)

All functions are designed to work with both BayesianMMM and extended model classes,
extracting data from traces and computing uncertainty-aware metrics.

Usage:
    from mmm_framework.reporting.helpers import (
        compute_roi_with_uncertainty,
        compute_channel_contributions,
        get_prior_posterior_comparison,
        compute_saturation_curves_with_uncertainty,
        compute_adstock_weights,
        compute_component_decomposition,
    )

    # After fitting a model
    roi_df = compute_roi_with_uncertainty(mmm, spend_data)
    prior_post = get_prior_posterior_comparison(mmm)
    sat_curves = compute_saturation_curves_with_uncertainty(mmm)
"""

# Protocols
from .protocols import HasModel, HasPanel, HasTrace

# Result containers
from .results import (
    AdstockResult,
    DecompositionResult,
    MediatedEffectResult,
    PriorPosteriorComparison,
    ROIResult,
    SaturationCurveResult,
)

# Utility functions
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
    _get_scaling_params,
    _get_trace,
    _safe_get_column,
    _safe_to_numpy,
    safe_get_samples,
)

# ROI computation
from .roi import (
    _extract_spend_from_model,
    _get_contribution_samples,
    compute_marginal_roi,
    compute_roi_with_uncertainty,
)

# Prior/posterior comparison
from .prior_posterior import (
    _select_key_parameters,
    compute_shrinkage_summary,
    get_prior_posterior_comparison,
)

# Saturation curves
from .saturation import (
    _apply_saturation,
    _get_beta_samples,
    _get_saturation_params,
    compute_saturation_curves_with_uncertainty,
)

# Adstock weights
from .adstock import (
    _get_adstock_alpha,
    _get_adstock_lmax,
    compute_adstock_weights,
)

# Component decomposition
from .decomposition import (
    _compute_decomposition_from_trace,
    _convert_model_decomposition,
    compute_component_decomposition,
    compute_decomposition_waterfall,
)

# Extended model helpers (mediated/cross effects)
from .mediated import (
    _compute_mediation_from_trace,
    _convert_mediation_df,
    compute_cross_effects,
    compute_mediated_effects,
)

# Summary generation
from .summary import (
    _get_diagnostics,
    _get_model_info,
    debug_posterior_structure,
    generate_model_summary,
)

__all__ = [
    # Protocols
    "HasTrace",
    "HasModel",
    "HasPanel",
    # Result containers
    "ROIResult",
    "PriorPosteriorComparison",
    "SaturationCurveResult",
    "AdstockResult",
    "DecompositionResult",
    "MediatedEffectResult",
    # Utility functions
    "_safe_to_numpy",
    "safe_get_samples",
    "_compute_hdi",
    "_get_trace",
    "_get_posterior",
    "_get_channel_names",
    "_get_scaling_params",
    "_flatten_samples",
    "_check_model_fitted",
    "_safe_get_column",
    # ROI functions
    "compute_roi_with_uncertainty",
    "compute_marginal_roi",
    "_extract_spend_from_model",
    "_get_contribution_samples",
    # Prior/posterior comparison
    "get_prior_posterior_comparison",
    "compute_shrinkage_summary",
    "_select_key_parameters",
    # Saturation
    "compute_saturation_curves_with_uncertainty",
    "_get_saturation_params",
    "_get_beta_samples",
    "_apply_saturation",
    # Adstock
    "compute_adstock_weights",
    "_get_adstock_alpha",
    "_get_adstock_lmax",
    # Decomposition
    "compute_component_decomposition",
    "compute_decomposition_waterfall",
    "_convert_model_decomposition",
    "_compute_decomposition_from_trace",
    # Extended models
    "compute_mediated_effects",
    "compute_cross_effects",
    "_convert_mediation_df",
    "_compute_mediation_from_trace",
    # Summary
    "debug_posterior_structure",
    "generate_model_summary",
    "_get_model_info",
    "_get_diagnostics",
]
