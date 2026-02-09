"""
Summary report generation functions for MMM reporting.

Functions for generating comprehensive model summaries with diagnostics.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    import arviz as az
except ImportError:
    az = None

from .adstock import compute_adstock_weights
from .decomposition import compute_component_decomposition
from .roi import compute_roi_with_uncertainty
from .saturation import compute_saturation_curves_with_uncertainty
from .utils import (
    _check_model_fitted,
    _get_channel_names,
    _get_trace,
)


def debug_posterior_structure(mmm):
    """Print posterior structure for debugging."""
    posterior = mmm._trace.posterior

    print("\n" + "=" * 60)
    print("POSTERIOR STRUCTURE DEBUG")
    print("=" * 60)

    for var_name in list(posterior.data_vars):
        da = posterior[var_name]
        print(f"{var_name}: dims={da.dims}, shape={da.shape}")

    # Test the exact operation that's failing
    print("\n" + "-" * 40)
    print("Testing problematic operations:")
    print("-" * 40)

    if "channel_contributions" in posterior:
        da = posterior["channel_contributions"]
        print(f"\nchannel_contributions: dims={da.dims}, shape={da.shape}")

        # The error (slice(None, None, None), 0) means [:, 0] on xarray
        # Let's test what works

        print("\nTest 1: da.values first, then index...")
        try:
            arr = da.values
            result = arr[:, 0]
            print(f"  SUCCESS: shape={result.shape}")
        except Exception as e:
            print(f"  FAILED: {e}")

        print("\nTest 2: Direct indexing on DataArray...")
        try:
            result = da[:, 0]
            print(f"  SUCCESS: {type(result)}")
        except Exception as e:
            print(f"  FAILED: {e}")
            print("  ^ THIS IS LIKELY YOUR BUG!")


def generate_model_summary(
    model: Any,
    hdi_prob: float = 0.94,
) -> dict[str, Any]:
    """
    Generate comprehensive model summary for reporting.

    Aggregates key metrics into a single dictionary suitable for
    report generation or dashboard display.

    Parameters
    ----------
    model : BayesianMMM or ExtendedMMM
        Fitted model
    hdi_prob : float
        HDI probability

    Returns
    -------
    dict
        Summary containing:
        - model_info: Basic model metadata
        - diagnostics: MCMC convergence diagnostics
        - roi_summary: ROI by channel
        - decomposition: Component contributions
        - saturation_summary: Saturation levels
        - adstock_summary: Carryover effects
    """
    _check_model_fitted(model)

    summary = {
        "model_info": _get_model_info(model),
        "diagnostics": _get_diagnostics(model),
    }

    # ROI
    try:
        roi_df = compute_roi_with_uncertainty(model, hdi_prob=hdi_prob)
        summary["roi_summary"] = roi_df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"ROI computation failed: {e}")
        logger.exception("ROI error traceback:")
        logger.debug("ROI error traceback:", exc_info=True)
        summary["roi_summary"] = None

    # Decomposition
    try:
        decomp = compute_component_decomposition(
            model, include_time_series=False, hdi_prob=hdi_prob
        )
        summary["decomposition"] = [d.to_dict() for d in decomp]
    except Exception as e:
        logger.warning(f"Decomposition failed: {e}")
        summary["decomposition"] = None

    # Saturation
    try:
        sat_curves = compute_saturation_curves_with_uncertainty(
            model, n_points=50, hdi_prob=hdi_prob
        )
        summary["saturation_summary"] = {
            ch: {
                "saturation_level": curve.saturation_level,
                "marginal_response": curve.marginal_response_at_current,
            }
            for ch, curve in sat_curves.items()
        }
    except Exception as e:
        logger.warning(f"Saturation computation failed: {e}")
        logger.exception("Saturation error traceback:")
        logger.debug("Saturation error traceback:", exc_info=True)
        summary["saturation_summary"] = None

    # Adstock
    try:
        adstock = compute_adstock_weights(model, hdi_prob=hdi_prob)
        summary["adstock_summary"] = {
            ch: {
                "half_life": result.half_life,
                "total_carryover": result.total_carryover,
                "alpha_mean": result.alpha_mean,
            }
            for ch, result in adstock.items()
        }
    except Exception as e:
        logger.warning(f"Adstock computation failed: {e}")
        summary["adstock_summary"] = None

    return summary


def _get_model_info(model: Any) -> dict[str, Any]:
    """Extract basic model info."""
    info = {
        "model_type": type(model).__name__,
        "n_obs": getattr(model, "n_obs", None),
        "n_channels": getattr(model, "n_channels", len(_get_channel_names(model))),
        "channel_names": _get_channel_names(model),
    }

    # Add geo/product info if available
    if hasattr(model, "has_geo"):
        info["has_geo"] = model.has_geo
        info["n_geos"] = getattr(model, "n_geos", None)

    if hasattr(model, "has_product"):
        info["has_product"] = model.has_product
        info["n_products"] = getattr(model, "n_products", None)

    # Extended model info
    if hasattr(model, "mediator_names"):
        info["mediator_names"] = list(model.mediator_names)
    if hasattr(model, "outcome_names"):
        info["outcome_names"] = list(model.outcome_names)

    return info


def _get_diagnostics(model: Any) -> dict[str, Any]:
    """Extract MCMC diagnostics."""
    trace = _get_trace(model)

    if trace is None:
        return {}

    diagnostics = {}

    try:
        if az is not None:
            summary = az.summary(trace)
            diagnostics["rhat_max"] = float(summary["r_hat"].max())
            diagnostics["ess_bulk_min"] = float(summary["ess_bulk"].min())
            diagnostics["ess_tail_min"] = float(summary["ess_tail"].min())

        # Check for divergences
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            diagnostics["divergences"] = int(
                trace.sample_stats["diverging"].values.sum()
            )
        else:
            diagnostics["divergences"] = 0

        # Convergence status
        diagnostics["converged"] = (
            diagnostics.get("divergences", 0) == 0
            and diagnostics.get("rhat_max", 2.0) < 1.01
            and diagnostics.get("ess_bulk_min", 0) > 400
        )

    except Exception as e:
        logger.warning(f"Error extracting diagnostics: {e}")

    return diagnostics


__all__ = [
    "debug_posterior_structure",
    "generate_model_summary",
    "_get_model_info",
    "_get_diagnostics",
]
