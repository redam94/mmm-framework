"""
Mixin classes providing shared extraction utilities.

These mixins provide reusable methods for data aggregation and
dimension-specific extraction (geo, product, time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .bundle import MMMDataBundle


class AggregationMixin:
    """
    Mixin providing data aggregation utilities for extractors.

    Provides methods for aggregating data by period, geography, and product
    while properly propagating uncertainty through sample aggregation.
    """

    def _aggregate_by_period_simple(
        self,
        values: np.ndarray,
        periods: list,
        unique_periods: list,
    ) -> np.ndarray:
        """
        Aggregate values by period using simple summation.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate, shape (n_obs,).
        periods : list
            Period label for each observation.
        unique_periods : list
            Unique periods in order.

        Returns
        -------
        np.ndarray
            Aggregated values, shape (n_periods,).
        """
        period_to_idx = {p: i for i, p in enumerate(unique_periods)}
        n_periods = len(unique_periods)

        result = np.zeros(n_periods)
        for i, (val, period) in enumerate(zip(values, periods)):
            if period in period_to_idx:
                result[period_to_idx[period]] += val

        return result

    def _aggregate_samples_by_period(
        self,
        samples: np.ndarray,
        periods: list,
        unique_periods: list,
        ci_prob: float = 0.8,
    ) -> dict[str, np.ndarray] | None:
        """
        Aggregate samples by period while preserving uncertainty.

        This method properly propagates uncertainty by aggregating
        samples first, then computing statistics on aggregated values.

        Parameters
        ----------
        samples : np.ndarray
            Sample array of shape (n_samples, n_obs).
        periods : list
            Period label for each observation.
        unique_periods : list
            Unique periods in order.
        ci_prob : float
            Credible interval probability.

        Returns
        -------
        dict[str, np.ndarray] or None
            Dictionary with "mean", "lower", "upper" arrays.
        """
        if samples is None or len(periods) == 0:
            return None

        try:
            period_to_idx = {p: i for i, p in enumerate(unique_periods)}
            obs_period_idx = np.array([period_to_idx.get(p, -1) for p in periods])

            n_samples = samples.shape[0]
            n_periods = len(unique_periods)

            # Aggregate samples by period
            samples_agg = np.zeros((n_samples, n_periods))

            for t in range(n_periods):
                mask = obs_period_idx == t
                if mask.any():
                    samples_agg[:, t] = samples[:, mask].sum(axis=1)

            # Compute statistics
            alpha = (1 - ci_prob) / 2

            return {
                "mean": samples_agg.mean(axis=0),
                "lower": np.percentile(samples_agg, alpha * 100, axis=0),
                "upper": np.percentile(samples_agg, (1 - alpha) * 100, axis=0),
            }

        except Exception:
            return None

    def _aggregate_by_group(
        self,
        values: np.ndarray,
        group_idx: np.ndarray,
        n_groups: int,
    ) -> np.ndarray:
        """
        Aggregate values by group index.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate.
        group_idx : np.ndarray
            Group index for each value.
        n_groups : int
            Number of groups.

        Returns
        -------
        np.ndarray
            Aggregated values per group.
        """
        result = np.zeros(n_groups)
        for i, val in enumerate(values):
            if 0 <= group_idx[i] < n_groups:
                result[group_idx[i]] += val
        return result

    def _aggregate_by_period(
        self,
        values: np.ndarray,
        n_periods: int,
        geo_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate values by period for a specific geo.

        If there are multiple products, this sums over products within each period.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate.
        n_periods : int
            Number of periods.
        geo_mask : np.ndarray
            Boolean mask for the geo (unused in simple case).

        Returns
        -------
        np.ndarray
            Aggregated values by period.
        """
        # Simple case: one observation per period per geo
        n_obs = len(values)
        if n_obs == n_periods:
            return values

        # Multiple products case: need to reshape and sum
        # Assumes data is ordered: periods are innermost, then geos, then products
        n_products = n_obs // n_periods
        if n_products * n_periods == n_obs:
            return values.reshape(n_products, n_periods).sum(axis=0)

        # Fallback: return as-is (may need custom handling)
        return values

    def _aggregate_by_period_with_indices(
        self,
        values: np.ndarray,
        time_idx: np.ndarray,
        dim_mask: np.ndarray,
        n_periods: int,
    ) -> np.ndarray:
        """
        Aggregate values by period using explicit time indices.

        This is a more robust aggregation method that uses the time_idx array
        to properly group observations by period, regardless of data ordering.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate, shape (n_obs,)
        time_idx : np.ndarray
            Time period index for each observation, shape (n_obs,)
        dim_mask : np.ndarray
            Boolean mask for filtering (e.g., specific geo or product), shape (n_obs,)
        n_periods : int
            Number of unique time periods

        Returns
        -------
        np.ndarray
            Aggregated values by period, shape (n_periods,)
        """
        result = np.zeros(n_periods)
        filtered_values = values[dim_mask]
        filtered_time_idx = time_idx[dim_mask]

        for t in range(n_periods):
            time_mask = filtered_time_idx == t
            if time_mask.any():
                result[t] = filtered_values[time_mask].sum()

        return result


class GeoExtractionMixin:
    """
    Mixin providing geo-level extraction methods.

    Requires the class to have `panel`, `mmm`, and `ci_prob` attributes.
    """

    def _get_geo_names(self) -> list[str] | None:
        """Get geography names from panel or model."""
        if hasattr(self.panel, "coords") and hasattr(self.panel.coords, "geographies"):
            geographies = self.panel.coords.geographies
            if geographies is not None:
                return list(geographies)
        if hasattr(self.mmm, "geo_names"):
            geo_names = self.mmm.geo_names
            if geo_names is not None:
                return list(geo_names)
        return None

    def _get_geo_indices(self) -> np.ndarray | None:
        """Get geo index for each observation."""
        if hasattr(self.mmm, "geo_idx") and self.mmm.geo_idx is not None:
            return np.array(self.mmm.geo_idx)
        if hasattr(self.panel, "geo_idx") and self.panel.geo_idx is not None:
            return np.array(self.panel.geo_idx)

        # Try to extract from panel's MultiIndex
        if self.panel is not None:
            geo_names = self._get_geo_names()
            if geo_names is not None and len(geo_names) > 1:
                # Try to get from y.index if it's a MultiIndex
                if hasattr(self.panel, "y") and hasattr(self.panel.y, "index"):
                    idx = self.panel.y.index
                    if hasattr(idx, "get_level_values"):
                        # Try both cases for geography level name
                        for level_name in ["geography", "Geography"]:
                            try:
                                geo_values = idx.get_level_values(level_name)
                                # Convert geo names to indices
                                geo_to_idx = {g: i for i, g in enumerate(geo_names)}
                                return np.array(
                                    [geo_to_idx.get(str(g), 0) for g in geo_values]
                                )
                            except KeyError:
                                continue
        return None

    def _get_time_indices(self) -> np.ndarray | None:
        """Get time index for each observation."""
        if hasattr(self.mmm, "time_idx") and self.mmm.time_idx is not None:
            return np.array(self.mmm.time_idx)

        # Try to extract from panel's MultiIndex
        if self.panel is not None:
            unique_periods = self._get_unique_periods()
            if unique_periods is not None and len(unique_periods) > 0:
                if hasattr(self.panel, "y") and hasattr(self.panel.y, "index"):
                    idx = self.panel.y.index
                    if hasattr(idx, "get_level_values"):
                        # Try both cases for period level name
                        for level_name in ["period", "Period"]:
                            try:
                                period_values = idx.get_level_values(level_name)
                                # Convert periods to indices
                                period_to_idx = {
                                    str(p): i for i, p in enumerate(unique_periods)
                                }
                                return np.array(
                                    [
                                        period_to_idx.get(str(p), 0)
                                        for p in period_values
                                    ]
                                )
                            except KeyError:
                                continue
                        # Try first level if period not found by name
                        try:
                            period_values = idx.get_level_values(0)
                            period_to_idx = {
                                str(p): i for i, p in enumerate(unique_periods)
                            }
                            return np.array(
                                [period_to_idx.get(str(p), 0) for p in period_values]
                            )
                        except Exception:
                            pass
                    else:
                        # Simple index - assume it's period only
                        period_to_idx = {
                            str(p): i for i, p in enumerate(unique_periods)
                        }
                        return np.array(
                            [period_to_idx.get(str(p), 0) for p in idx]
                        )

        # Fallback: construct from panel coords
        if self.panel is not None and hasattr(self.panel, "coords"):
            n_obs = len(self.panel.y)
            n_periods = self.panel.coords.n_periods
            n_geos = self.panel.coords.n_geos
            n_products = self.panel.coords.n_products
            # Assumes data ordered as periods innermost
            if n_obs == n_periods * n_geos * n_products:
                return np.tile(np.arange(n_periods), n_geos * n_products)
        return None

    def _get_actual_original_scale(self) -> np.ndarray | None:
        """Get observed values in original scale."""
        if self.panel is None:
            return None

        y_standardized = self.panel.y.values.flatten()
        y_mean = self.mmm.y_mean if hasattr(self.mmm, "y_mean") else 0
        y_std = self.mmm.y_std if hasattr(self.mmm, "y_std") else 1

        return y_standardized * y_std + y_mean

    def _get_predictions_original_scale(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """Get predictions in original scale with uncertainty."""
        trace = getattr(self.mmm, "_trace", None)
        if trace is None:
            return None, None, None

        try:
            if hasattr(trace, "posterior_predictive") and "y" in trace.posterior_predictive:
                y_samples = trace.posterior_predictive["y"].values
                y_samples = y_samples.reshape(-1, y_samples.shape[-1])
            elif hasattr(trace, "posterior") and "mu" in trace.posterior:
                y_samples = trace.posterior["mu"].values
                y_samples = y_samples.reshape(-1, y_samples.shape[-1])
            else:
                return None, None, None

            # Transform to original scale
            y_mean = self.mmm.y_mean if hasattr(self.mmm, "y_mean") else 0
            y_std = self.mmm.y_std if hasattr(self.mmm, "y_std") else 1

            y_samples_orig = y_samples * y_std + y_mean

            y_pred_mean = y_samples_orig.mean(axis=0)
            alpha = (1 - self.ci_prob) / 2
            y_pred_lower = np.percentile(y_samples_orig, alpha * 100, axis=0)
            y_pred_upper = np.percentile(y_samples_orig, (1 - alpha) * 100, axis=0)

            return y_pred_mean, y_pred_lower, y_pred_upper

        except Exception:
            return None, None, None


class ProductExtractionMixin:
    """
    Mixin providing product-level extraction methods.

    Requires the class to have `panel`, `mmm`, and `ci_prob` attributes.
    """

    def _get_product_names(self) -> list[str] | None:
        """Get product names from panel or model."""
        if hasattr(self.panel, "coords") and hasattr(self.panel.coords, "products"):
            products = self.panel.coords.products
            if products is not None:
                return list(products)
        if hasattr(self.mmm, "product_names"):
            return list(self.mmm.product_names)
        return None

    def _get_product_indices(self) -> np.ndarray | None:
        """Get product index for each observation."""
        if hasattr(self.mmm, "product_idx"):
            return np.array(self.mmm.product_idx)
        if hasattr(self.panel, "product_idx"):
            return np.array(self.panel.product_idx)
        return None


__all__ = [
    "AggregationMixin",
    "GeoExtractionMixin",
    "ProductExtractionMixin",
]
