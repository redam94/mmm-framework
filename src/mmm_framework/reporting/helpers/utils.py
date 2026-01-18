"""
Utility functions for MMM helper computations.

Low-level utility functions for data extraction, conversion, and statistical
computation used across the helpers module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

try:
    import arviz as az
except ImportError:
    az = None


def _safe_to_numpy(data: Any) -> np.ndarray | None:
    """Convert DataFrame or array-like to numpy array safely."""
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        return data

    if hasattr(data, "values"):
        return data.values

    try:
        return np.asarray(data)
    except Exception:
        return None


def safe_get_samples(posterior, var_name, channel_idx=None):
    """
    Safely extract samples from posterior, handling both xarray and numpy.

    This is a diagnostic wrapper to help identify indexing issues.
    """
    if var_name not in posterior:
        return None

    data = posterior[var_name]

    # Log what we're working with
    logger.debug(
        f"Extracting {var_name}: type={type(data)}, "
        f"hasattr values={hasattr(data, 'values')}, "
        f"hasattr dims={hasattr(data, 'dims')}"
    )

    # If it's an xarray DataArray, get the numpy array first
    if hasattr(data, "values"):
        logger.debug(
            f"  dims={getattr(data, 'dims', 'N/A')}, shape before .values: {data.shape}"
        )
        arr = data.values
    else:
        arr = data

    logger.debug(f"  After .values: type={type(arr)}, shape={arr.shape}, ndim={arr.ndim}")

    # Now do any indexing on the numpy array
    if channel_idx is not None and arr.ndim > 2:
        logger.debug(f"  Indexing with channel_idx={channel_idx}")
        # Flatten chain x draw first, then index channel
        n_chains, n_draws = arr.shape[0], arr.shape[1]
        arr = arr.reshape(n_chains * n_draws, *arr.shape[2:])
        if channel_idx < arr.shape[-1]:
            arr = arr[..., channel_idx]

    return arr


def _compute_hdi(
    samples: np.ndarray,
    prob: float = 0.94,
) -> tuple[float, float]:
    """
    Compute highest density interval from samples.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples (1D array)
    prob : float
        Probability mass for HDI (default 0.94)

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of HDI
    """
    samples = np.asarray(samples).flatten()
    samples = samples[~np.isnan(samples)]

    if len(samples) == 0:
        return np.nan, np.nan

    if az is not None:
        try:
            hdi = az.hdi(samples, hdi_prob=prob)
            return float(hdi[0]), float(hdi[1])
        except Exception:
            pass

    # Fallback to percentile-based interval
    alpha = (1 - prob) / 2
    return float(np.percentile(samples, alpha * 100)), float(
        np.percentile(samples, (1 - alpha) * 100)
    )


def _get_trace(model: Any) -> Any | None:
    """Extract ArviZ trace from model."""
    if hasattr(model, "_trace"):
        return model._trace
    if hasattr(model, "trace"):
        return model.trace
    return None


def _get_posterior(model: Any) -> Any | None:
    """Extract posterior from trace."""
    trace = _get_trace(model)
    if trace is not None and hasattr(trace, "posterior"):
        return trace.posterior
    return None


def _get_channel_names(model: Any) -> list[str]:
    """Extract channel names from model."""
    if hasattr(model, "channel_names"):
        return list(model.channel_names)
    if hasattr(model, "panel") and model.panel is not None:
        if hasattr(model.panel, "channel_names"):
            return list(model.panel.channel_names)
    return []


def _get_scaling_params(model: Any) -> tuple[float, float]:
    """Get y_mean and y_std from model for rescaling."""
    y_mean = getattr(model, "y_mean", 0.0)
    y_std = getattr(model, "y_std", 1.0)
    return float(y_mean), float(y_std)


def _flatten_samples(data) -> np.ndarray:
    """Flatten chain and draw dimensions from posterior samples."""
    # FIRST: Convert to numpy if it's xarray
    if hasattr(data, "values"):
        arr = data.values
    else:
        arr = np.asarray(data)

    # THEN: Flatten
    if arr.ndim >= 2:
        return arr.reshape(-1, *arr.shape[2:]) if arr.ndim > 2 else arr.flatten()
    return arr.flatten()


def _check_model_fitted(model: Any) -> None:
    """Raise error if model is not fitted."""
    trace = _get_trace(model)
    if trace is None:
        raise ValueError("Model not fitted. Call fit() first.")


def _safe_get_column(
    data: Any, col_idx: int, col_name: str = None
) -> np.ndarray | None:
    """
    Safely extract a column from DataFrame or numpy array.

    Handles both DataFrame (needs .iloc or column name) and numpy array (needs [:, idx]).
    """
    if data is None:
        return None

    try:
        # If it's a DataFrame with the column name
        if (
            col_name is not None
            and hasattr(data, "columns")
            and col_name in data.columns
        ):
            return data[col_name].values

        # If it's a DataFrame, use .iloc
        if hasattr(data, "iloc"):
            if col_idx < data.shape[1]:
                return data.iloc[:, col_idx].values

        # If it has .values (DataFrame-like), convert first
        elif hasattr(data, "values"):
            arr = data.values
            if col_idx < arr.shape[1]:
                return arr[:, col_idx]

        # Plain numpy array
        else:
            if col_idx < data.shape[1]:
                return data[:, col_idx]

    except Exception as e:
        logger.debug(
            f"_safe_get_column failed for col_idx={col_idx}, col_name={col_name}: {e}"
        )

    return None


__all__ = [
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
]
