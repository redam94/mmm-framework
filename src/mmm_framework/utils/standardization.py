"""Data standardization utilities for MMM Framework.

This module provides utilities for standardizing data (zero mean, unit variance)
which is commonly needed for Bayesian models to ensure numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class StandardizationParams:
    """Parameters from standardization fit.

    Stores the mean and standard deviation used for standardization,
    allowing the transformation to be applied to new data or reversed.

    Attributes
    ----------
    mean : float | NDArray
        Mean value(s) used for centering. Scalar for 1D data,
        array for multi-dimensional data.
    std : float | NDArray
        Standard deviation(s) used for scaling. Scalar for 1D data,
        array for multi-dimensional data.
    """

    mean: float | NDArray
    std: float | NDArray

    def to_dict(self) -> dict:
        """Convert to serializable dictionary.

        Returns
        -------
        dict
            Dictionary with 'mean' and 'std' keys, with numpy arrays
            converted to lists for JSON serialization.
        """
        return {
            "mean": float(self.mean) if np.isscalar(self.mean) or self.mean.ndim == 0 else self.mean.tolist(),
            "std": float(self.std) if np.isscalar(self.std) or self.std.ndim == 0 else self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> StandardizationParams:
        """Create from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with 'mean' and 'std' keys.

        Returns
        -------
        StandardizationParams
            Reconstructed parameters object.
        """
        mean = np.array(d["mean"]) if isinstance(d["mean"], list) else d["mean"]
        std = np.array(d["std"]) if isinstance(d["std"], list) else d["std"]
        return cls(mean=mean, std=std)


class DataStandardizer:
    """Standardize data with zero mean and unit variance.

    This class provides methods for standardizing data (z-score normalization)
    which is essential for Bayesian models. It handles both 1D and 2D data,
    and includes a small epsilon term to prevent division by zero for
    constant data.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant added to standard deviation to prevent division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.utils import DataStandardizer
    >>>
    >>> # Create standardizer
    >>> standardizer = DataStandardizer()
    >>>
    >>> # Fit and transform training data
    >>> data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> standardized, params = standardizer.fit_transform(data)
    >>>
    >>> # Transform new data using same parameters
    >>> new_data = np.array([25.0, 35.0])
    >>> transformed = standardizer.transform(new_data, params)
    >>>
    >>> # Reverse transformation
    >>> original_scale = standardizer.inverse_transform(transformed, params)
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initialize DataStandardizer.

        Parameters
        ----------
        epsilon : float, default=1e-8
            Small constant added to standard deviation to prevent
            division by zero when data has zero variance.
        """
        self.epsilon = epsilon
        self._params: StandardizationParams | None = None

    def fit(self, data: NDArray) -> StandardizationParams:
        """Compute standardization parameters from data.

        Parameters
        ----------
        data : NDArray
            Input data to compute parameters from. Can be 1D or 2D.
            For 2D data, parameters are computed per column (axis=0).

        Returns
        -------
        StandardizationParams
            Parameters containing mean and standard deviation.
        """
        mean = data.mean(axis=0)
        std = data.std(axis=0) + self.epsilon
        self._params = StandardizationParams(mean=mean, std=std)
        return self._params

    def transform(
        self,
        data: NDArray,
        params: StandardizationParams | None = None,
    ) -> NDArray:
        """Apply standardization to data.

        Parameters
        ----------
        data : NDArray
            Data to standardize.
        params : StandardizationParams, optional
            Parameters to use for transformation. If None, uses parameters
            from most recent fit() call.

        Returns
        -------
        NDArray
            Standardized data with zero mean and unit variance.

        Raises
        ------
        ValueError
            If params is None and fit() has not been called.
        """
        p = params or self._params
        if p is None:
            raise ValueError("Must call fit() first or provide params")
        return (data - p.mean) / p.std

    def fit_transform(self, data: NDArray) -> tuple[NDArray, StandardizationParams]:
        """Fit parameters and transform data in one step.

        Parameters
        ----------
        data : NDArray
            Data to fit and transform.

        Returns
        -------
        tuple[NDArray, StandardizationParams]
            Tuple of (standardized_data, parameters).
        """
        params = self.fit(data)
        return self.transform(data, params), params

    def inverse_transform(
        self,
        data: NDArray,
        params: StandardizationParams | None = None,
    ) -> NDArray:
        """Reverse standardization to recover original scale.

        Parameters
        ----------
        data : NDArray
            Standardized data to transform back.
        params : StandardizationParams, optional
            Parameters to use for inverse transformation. If None, uses
            parameters from most recent fit() call.

        Returns
        -------
        NDArray
            Data in original scale.

        Raises
        ------
        ValueError
            If params is None and fit() has not been called.
        """
        p = params or self._params
        if p is None:
            raise ValueError("Must call fit() first or provide params")
        return data * p.std + p.mean
