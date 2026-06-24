"""
Mixin classes providing shared extraction utilities.

These mixins provide reusable methods for data aggregation and
dimension-specific extraction (geo, product, time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from .bundle import MMMDataBundle


def _finite(x: Any) -> float | None:
    """Coerce to a finite float, or ``None`` (for missing / non-finite values)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


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
                        return np.array([period_to_idx.get(str(p), 0) for p in idx])

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
            if (
                hasattr(trace, "posterior_predictive")
                and "y" in trace.posterior_predictive
            ):
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


class EstimandPPCMixin:
    """Shared extraction of estimand results (mean + CI) and the
    posterior-predictive goodness-of-fit summary.

    Both ``BayesianMMMExtractor`` and ``ExtendedMMMExtractor`` inherit this. Two
    hooks adapt it per model family:

    - :meth:`_estimand_model` — the model object exposing ``evaluate_estimands``
      (defaults to ``self.mmm`` or ``self.model``).
    - :meth:`_ppc_arrays` — returns aligned, original-scale
      ``(observed, y_rep, pred_mean, pred_lower, pred_upper)`` posterior-predictive
      arrays, or ``(None,)*5`` when the model cannot produce them. Each extractor
      implements this differently (a core MMM resamples via ``predict``; an
      extended model samples its PyMC graph).

    Both ``_extract_*`` methods are best-effort: any failure leaves the bundle
    field empty so the corresponding section simply no-ops.
    """

    #: Nominal interval widths sampled for the calibration curve.
    _PPC_NOMINAL_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    #: Number of replicate density curves kept for the overlay chart.
    _PPC_OVERLAY_CURVES = 40

    # -- hooks ----------------------------------------------------------------

    def _estimand_model(self) -> Any:
        """The model object exposing ``evaluate_estimands`` (family-specific)."""
        return getattr(self, "mmm", None) or getattr(self, "model", None)

    def _ppc_arrays(self):
        """Aligned ``(observed, y_rep, pred_mean, pred_lower, pred_upper)`` in
        original KPI scale at the observation level, or ``(None,)*5``. Subclasses
        override; the default produces nothing."""
        return (None, None, None, None, None)

    # -- estimands ------------------------------------------------------------

    def _extract_estimands(self, bundle: "MMMDataBundle") -> "MMMDataBundle":
        """Realize the model's declared / default estimands as mean + CI.

        Prefers estimands already realized on the fit results (computed at fit
        time when the model declares them); otherwise evaluates them on demand via
        ``model.evaluate_estimands()``. A model that does not implement the
        estimand engine (e.g. an extended model without ``evaluate_estimands``)
        leaves ``bundle.estimands`` empty.
        """
        try:
            results = getattr(self.results, "estimands", None) if self.results else None
            if not results:
                model = self._estimand_model()
                fn = getattr(model, "evaluate_estimands", None)
                if not callable(fn):
                    return bundle
                results = fn()
            if not results:
                return bundle

            out: dict[str, dict[str, Any]] = {}
            for key, r in results.items():
                mean = _finite(getattr(r, "mean", None))
                if getattr(r, "status", "ok") != "ok" or mean is None:
                    continue
                entry: dict[str, Any] = {
                    "mean": mean,
                    "lower": _finite(getattr(r, "hdi_low", None)),
                    "upper": _finite(getattr(r, "hdi_high", None)),
                    "kind": getattr(r, "kind", "") or "",
                    "units": getattr(r, "units", "") or "",
                    "hdi_prob": float(getattr(r, "hdi_prob", 0.94) or 0.94),
                }
                extra = getattr(r, "extra", None) or {}
                for k in ("contribution_pct", "prob_positive", "prob_profitable"):
                    val = _finite(extra.get(k)) if isinstance(extra, dict) else None
                    if val is not None:
                        entry[k] = val
                out[str(key)] = entry

            if out:
                bundle.estimands = out
        except Exception:  # noqa: BLE001 — reporting must never hard-fail
            logger.debug("estimands extraction skipped", exc_info=True)
        return bundle

    # -- posterior-predictive goodness-of-fit ---------------------------------

    def _extract_posterior_predictive(self, bundle: "MMMDataBundle") -> "MMMDataBundle":
        """Compute the posterior-predictive goodness-of-fit summary.

        Pulls aligned posterior-predictive replicates (original KPI scale,
        observation level) via :meth:`_ppc_arrays`, then derives the compact,
        chart-ready summary the PosteriorPredictiveSection consumes: per-obs mean
        + interval, a down-sampled replicate cloud, an interval-calibration curve,
        and posterior-predictive p-values for summary statistics. Best-effort.
        """
        try:
            observed, y_rep, pred_mean, pred_lower, pred_upper = self._ppc_arrays()
            if observed is None or y_rep is None:
                return bundle

            ci = float(self.ci_prob)
            coverage = []
            for p in self._PPC_NOMINAL_LEVELS:
                lo = np.percentile(y_rep, (1 - p) / 2 * 100, axis=0)
                hi = np.percentile(y_rep, (1 + p) / 2 * 100, axis=0)
                emp = float(np.mean((observed >= lo) & (observed <= hi)))
                coverage.append({"nominal": float(p), "empirical": emp})

            ss_res = float(np.sum((observed - pred_mean) ** 2))
            ss_tot = float(np.sum((observed - observed.mean()) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None

            bundle.posterior_predictive = {
                "observed": observed,
                "pred_mean": pred_mean,
                "pred_lower": pred_lower,
                "pred_upper": pred_upper,
                "samples": self._downsample_rows(y_rep, self._PPC_OVERLAY_CURVES),
                "coverage": coverage,
                "bayes_p": self._ppc_bayes_p(observed, y_rep),
                "ci_level": ci,
                "r2": r2,
            }
        except Exception:  # noqa: BLE001 — reporting must never hard-fail
            logger.debug("posterior-predictive extraction skipped", exc_info=True)
        return bundle

    @staticmethod
    def _ppc_bayes_p(observed: np.ndarray, y_rep: np.ndarray) -> dict[str, float]:
        """Posterior-predictive p-values: P(T(y_rep) >= T(y_obs)) per statistic."""
        stat_fns = {
            "mean": lambda a, axis=None: np.mean(a, axis=axis),
            "std": lambda a, axis=None: np.std(a, axis=axis),
            "min": lambda a, axis=None: np.min(a, axis=axis),
            "max": lambda a, axis=None: np.max(a, axis=axis),
        }
        out: dict[str, float] = {}
        for name, fn in stat_fns.items():
            obs_stat = float(fn(observed))
            rep_stat = fn(y_rep, axis=1)
            out[name] = float(np.mean(rep_stat >= obs_stat))
        return out

    @staticmethod
    def _downsample_rows(arr: np.ndarray, n: int) -> np.ndarray:
        """Deterministically thin rows to at most ``n`` (evenly spaced)."""
        if arr.shape[0] <= n:
            return arr
        idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
        return arr[idx]


__all__ = [
    "AggregationMixin",
    "GeoExtractionMixin",
    "ProductExtractionMixin",
    "EstimandPPCMixin",
    "_finite",
]
