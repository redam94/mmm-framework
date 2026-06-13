"""
ExtendedMMMExtractor - Extract data from mmm-framework's extended MMM models.

Supports NestedMMM, MultivariateMMM, and CombinedMMM.

Scaling convention: the extension models standardize outcomes internally for
sampling, but register report-consumed deterministics (``mu``,
``effect_<med>_on_y``, ``direct_effect_<ch>``, ``indirect_*``, ``total_*``)
in ORIGINAL units. Free RVs (``delta_direct_*``, ``beta_media``, ``alpha``/
``alpha_y`` intercepts) remain on the standardized scale and are converted
here via the model's ``y_mean``/``y_std`` (or per-outcome
``outcome_means``/``outcome_stds``). Models saved before internal
standardization lack those attributes and default to (0, 1) — their
deterministics were already raw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ...transforms.adstock import adstock_weights, apply_adstock
from .base import DataExtractor
from .bundle import MMMDataBundle

# Adstock kernel length used by the extension models' media transform. Mirrors
# mmm_extensions.models.base._ADSTOCK_L_MAX (not imported here to keep PyMC out
# of the reporting import path).
_ADSTOCK_L_MAX = 8


class ExtendedMMMExtractor(DataExtractor):
    """
    Extract data from mmm-framework's extended MMM models.

    Inherits shared utilities from DataExtractor for HDI computation,
    fit statistics, and MCMC diagnostics.

    Supports NestedMMM (mediation pathways), MultivariateMMM (per-outcome fit,
    decomposition, and cross-effects), and CombinedMMM (both).

    Parameters
    ----------
    model : Any
        Extended MMM model instance (NestedMMM, MultivariateMMM, or CombinedMMM)
    panel : Any, optional
        Accepted for interface parity with BayesianMMMExtractor (the extension
        models carry their own data arrays).
    results : Any, optional
        Fit results, if available.
    ci_prob : float
        Credible interval probability (default 0.8)
    """

    def __init__(
        self,
        model: Any,
        panel: Any | None = None,
        results: Any | None = None,
        ci_prob: float = 0.8,
    ):
        self.model = model
        self.panel = panel
        self.results = results
        self._ci_prob = ci_prob

    @property
    def ci_prob(self) -> float:
        """Credible interval probability."""
        return self._ci_prob

    # ------------------------------------------------------------------
    # Top level
    # ------------------------------------------------------------------

    def extract(self) -> MMMDataBundle:
        """Extract data from extended MMM model."""
        bundle = MMMDataBundle()

        bundle.channel_names = self._get_channel_names()
        bundle.dates = self._get_dates()
        bundle.actual = self._get_actual()

        is_multi = bool(getattr(self.model, "outcome_names", None))
        is_nested = bool(getattr(self.model, "mediator_names", None))

        posterior = self._posterior
        if posterior is not None:
            bundle.diagnostics = self._extract_diagnostics(self.model._trace)

            if is_multi:
                self._extract_multioutcome_fit(bundle)
                self._extract_multioutcome_decomposition(bundle)
                self._extract_cross_effects(bundle)
            else:
                self._extract_univariate_fit(bundle)
                if is_nested:
                    self._extract_nested_decomposition(bundle)

            if is_nested:
                self._extract_mediation(bundle)

            bundle.channel_roi = self._compute_channel_roi()
            bundle.saturation_curves = self._get_saturation_curves()
            bundle.adstock_curves = self._get_adstock_curves()
            bundle.current_spend = self._get_current_spend()

            if bundle.actual is not None:
                bundle.total_revenue = float(np.asarray(bundle.actual).sum())

            self._extract_summary_metrics(bundle)

        bundle.model_specification = self._get_model_specification()

        return bundle

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def _posterior(self):
        trace = getattr(self.model, "_trace", None)
        if trace is None or not hasattr(trace, "posterior"):
            return None
        return trace.posterior

    def _samples(self, name: str) -> np.ndarray | None:
        """Flattened posterior samples of a scalar variable, or None."""
        posterior = self._posterior
        if posterior is None or name not in posterior:
            return None
        return np.asarray(posterior[name].values, dtype=float).reshape(-1)

    def _mean(self, name: str) -> float | None:
        samples = self._samples(name)
        return float(samples.mean()) if samples is not None else None

    def _obs_mean(self, name: str) -> np.ndarray | None:
        """Posterior mean of a per-obs deterministic, or None."""
        posterior = self._posterior
        if posterior is None or name not in posterior:
            return None
        vals = np.asarray(posterior[name].values, dtype=float)
        return vals.mean(axis=(0, 1))

    def _stats(self, samples: np.ndarray) -> dict[str, float]:
        samples = np.asarray(samples, dtype=float).reshape(-1)
        lower, upper = self._compute_percentile_bounds(samples)
        return {
            "mean": float(samples.mean()),
            "lower": float(lower),
            "upper": float(upper),
        }

    def _predicted_from_samples(self, samples: np.ndarray) -> dict[str, np.ndarray]:
        """Build {"mean","lower","upper"} from (n_samples, n_obs) draws."""
        lower, upper = self._compute_percentile_bounds(samples, axis=0)
        return {"mean": samples.mean(axis=0), "lower": lower, "upper": upper}

    def _get_channel_names(self) -> list[str]:
        """Get channel names from model."""
        if hasattr(self.model, "channel_names"):
            return list(self.model.channel_names)
        return []

    def _get_dates(self) -> np.ndarray | None:
        """Get date index from model."""
        if hasattr(self.model, "index"):
            return np.array(self.model.index)
        return None

    def _get_actual(self) -> np.ndarray | None:
        """Actual KPI values (primary outcome for multi-outcome models)."""
        outcome_data = getattr(self.model, "outcome_data", None)
        outcome_names = getattr(self.model, "outcome_names", None)
        if outcome_data and outcome_names:
            return np.asarray(outcome_data[outcome_names[0]], dtype=float)
        if hasattr(self.model, "y"):
            return np.asarray(self.model.y, dtype=float)
        return None

    def _media_x_sat(self, ch_idx: int, channel: str) -> np.ndarray | None:
        """Re-evaluate a channel's media transform at posterior-mean parameters.

        Mirrors BaseExtendedMMM._media_transform_apply: normalize by the
        per-channel raw max, geometric adstock (normalized kernel), then
        logistic saturation ``1 - exp(-lam * x)``.
        """
        X = np.asarray(self.model.X_media, dtype=float)
        scales = getattr(self.model, "_media_scale", None)
        if scales is None:
            scales = X.max(axis=0) + 1e-8
        scale = float(scales[ch_idx])

        alpha = self._mean(f"alpha_{channel}")
        if alpha is None:
            alpha = self._mean("alpha_shared")
        lam = self._mean(f"lambda_{channel}")
        if alpha is None or lam is None:
            return None

        weights = adstock_weights(
            "geometric", _ADSTOCK_L_MAX, alpha=float(np.clip(alpha, 1e-6, 1 - 1e-6))
        )
        x_ad = apply_adstock(X[:, ch_idx] / scale, weights)
        return 1.0 - np.exp(-lam * x_ad)

    def _outcome_scale(self, outcome: str | None = None) -> tuple[float, float]:
        """(mean, std) of the (primary) outcome's standardization, with
        raw-scale (pre-standardization era) models defaulting to (0, 1)."""
        outcome_names = getattr(self.model, "outcome_names", None)
        if outcome_names:
            outcome = outcome or outcome_names[0]
            means = getattr(self.model, "outcome_means", None) or {}
            stds = getattr(self.model, "outcome_stds", None) or {}
            return float(means.get(outcome, 0.0)), float(stds.get(outcome, 1.0))
        return (
            float(getattr(self.model, "y_mean", 0.0)),
            float(getattr(self.model, "y_std", 1.0)),
        )

    def _total_effect_samples(self, channel: str) -> np.ndarray | None:
        """Posterior samples of a channel's total effect coefficient on the
        (primary) outcome, in original KPI units per unit saturated spend:
        direct plus all mediated paths."""
        posterior = self._posterior
        if posterior is None:
            return None

        outcome_names = getattr(self.model, "outcome_names", None)
        if outcome_names:
            # CombinedMMM registers total_<ch>_<outcome> in original units;
            # MultivariateMMM has standardized beta_media (outcome, channel).
            samples = self._samples(f"total_{channel}_{outcome_names[0]}")
            if samples is not None:
                return samples
            if "beta_media" in posterior:
                vals = np.asarray(posterior["beta_media"].values, dtype=float)
                ch_idx = self._get_channel_names().index(channel)
                _, std = self._outcome_scale()
                return vals[:, :, 0, ch_idx].reshape(-1) * std
            return None

        _, y_std = self._outcome_scale()
        total = None
        direct = self._samples(f"delta_direct_{channel}")
        if direct is not None:
            total = direct * y_std
        for med, indirect in self._indirect_effect_samples(channel).items():
            total = indirect if total is None else total + indirect
        return total

    # ------------------------------------------------------------------
    # Fit (actual vs predicted)
    # ------------------------------------------------------------------

    def _extract_univariate_fit(self, bundle: MMMDataBundle) -> None:
        """Fit data for single-outcome models (NestedMMM) from ``mu``."""
        posterior = self._posterior
        if posterior is None or "mu" not in posterior:
            return
        try:
            vals = np.asarray(posterior["mu"].values, dtype=float)
            samples = vals.reshape(-1, vals.shape[-1])
            bundle.predicted = self._predicted_from_samples(samples)
            bundle.fit_statistics = self._compute_fit_statistics(
                bundle.actual, bundle.predicted
            )
        except Exception as e:
            logger.warning(f"Extended fit extraction failed: {e}")

    def _extract_multioutcome_fit(self, bundle: MMMDataBundle) -> None:
        """Per-outcome fit for MultivariateMMM/CombinedMMM.

        Outcomes are exposed through the bundle's product dimension, which
        drives the per-outcome selector in the fit section.
        """
        posterior = self._posterior
        outcome_names = list(self.model.outcome_names)
        if posterior is None or "mu" not in posterior:
            return
        try:
            vals = np.asarray(posterior["mu"].values, dtype=float)
            # (chain, draw, obs, outcome) -> (samples, obs, outcome)
            samples = vals.reshape(-1, *vals.shape[2:])

            bundle.product_names = outcome_names
            bundle.actual_by_product = {}
            bundle.predicted_by_product = {}
            bundle.fit_statistics_by_product = {}

            for k, name in enumerate(outcome_names):
                actual_k = np.asarray(self.model.outcome_data[name], dtype=float)
                predicted_k = self._predicted_from_samples(samples[:, :, k])
                bundle.actual_by_product[name] = actual_k
                bundle.predicted_by_product[name] = predicted_k
                bundle.fit_statistics_by_product[name] = self._compute_fit_statistics(
                    actual_k, predicted_k
                )

            # Primary outcome doubles as the aggregate view
            primary = outcome_names[0]
            bundle.actual = bundle.actual_by_product[primary]
            bundle.predicted = bundle.predicted_by_product[primary]
            bundle.fit_statistics = bundle.fit_statistics_by_product[primary]
        except Exception as e:
            logger.warning(f"Extended multi-outcome fit extraction failed: {e}")

    # ------------------------------------------------------------------
    # Mediation (NestedMMM / CombinedMMM)
    # ------------------------------------------------------------------

    def _affecting_channels(self, mediator: str) -> list[str]:
        try:
            return list(self.model._get_affecting_channels(mediator))
        except Exception:
            return self._get_channel_names()

    def _indirect_effect_samples(self, channel: str) -> dict[str, np.ndarray]:
        """Posterior samples of channel -> mediator -> outcome coefficients.

        Prefers the registered ``indirect_<ch>_via_<med>`` deterministics
        (multi-channel mediators; already in original units); reconstructs
        ``beta * gamma * y_std`` for single-channel mediators where that
        deterministic is absent.
        """
        out: dict[str, np.ndarray] = {}
        _, y_std = self._outcome_scale()
        for med in getattr(self.model, "mediator_names", []) or []:
            samples = self._samples(f"indirect_{channel}_via_{med}")
            if samples is None:
                gamma = self._samples(f"gamma_{med}")
                beta = self._samples(f"beta_{channel}_to_{med}")
                if beta is None and self._affecting_channels(med) == [channel]:
                    beta = self._samples(f"beta_media_to_{med}")
                if gamma is not None and beta is not None:
                    samples = beta * gamma * y_std
            if samples is not None:
                out[med] = samples
        return out

    def _extract_mediation(self, bundle: MMMDataBundle) -> None:
        """Mediator pathways, time series, and total indirect share."""
        posterior = self._posterior
        if posterior is None:
            return
        try:
            mediators = list(self.model.mediator_names)
            channels = self._get_channel_names()
            outcome_names = getattr(self.model, "outcome_names", None)
            primary = outcome_names[0] if outcome_names else None

            bundle.mediator_names = mediators

            pathways: dict[str, dict[str, Any]] = {}
            mediator_effects: dict[str, dict[str, float]] = {}
            all_direct: list[np.ndarray] = []
            all_indirect: list[np.ndarray] = []

            for ch in channels:
                entry: dict[str, Any] = {}

                if primary is not None:
                    # CombinedMMM: per-(channel, outcome) coefficients
                    direct = self._samples(f"direct_{ch}_{primary}")
                    indirect = self._samples(f"indirect_{ch}_{primary}")
                    total = self._samples(f"total_{ch}_{primary}")
                    per_med = {}
                else:
                    direct = self._samples(f"delta_direct_{ch}")
                    if direct is not None:
                        _, y_std = self._outcome_scale()
                        direct = direct * y_std
                    per_med = self._indirect_effect_samples(ch)
                    indirect = None
                    for med, s in per_med.items():
                        indirect = s if indirect is None else indirect + s
                    total = None

                n = None
                for s in (direct, indirect):
                    if s is not None:
                        n = len(s)
                        break
                if n is None:
                    continue
                if direct is None:
                    direct = np.zeros(n)
                if indirect is None:
                    indirect = np.zeros(n)
                if total is None:
                    total = direct + indirect

                entry["_direct"] = self._stats(direct)
                entry["_indirect"] = self._stats(indirect)
                entry["_total"] = self._stats(total)
                for med, s in per_med.items():
                    entry[med] = self._stats(s)
                    mediator_effects[f"{ch} → {med}"] = self._stats(s)

                pathways[ch] = entry
                all_direct.append(direct)
                all_indirect.append(indirect)

            if pathways:
                bundle.mediator_pathways = pathways
            if mediator_effects:
                bundle.mediator_effects = mediator_effects

            # Share of total marketing effect flowing through mediators
            if all_direct and all_indirect:
                direct_sum = np.sum(all_direct, axis=0)
                indirect_sum = np.sum(all_indirect, axis=0)
                total_sum = direct_sum + indirect_sum
                ok = np.abs(total_sum) > 1e-12
                if ok.any():
                    proportion = indirect_sum[ok] / total_sum[ok]
                    bundle.total_indirect_effect = self._stats(proportion)

            # Mediator latent values over time
            ts: dict[str, np.ndarray] = {}
            for med in mediators:
                latent = self._obs_mean(f"{med}_latent")
                if latent is not None:
                    ts[med] = latent
            if ts:
                bundle.mediator_time_series = ts
        except Exception as e:
            logger.warning(f"Mediation extraction failed: {e}")

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def _extract_nested_decomposition(self, bundle: MMMDataBundle) -> None:
        """Component time series for NestedMMM: baseline, per-mediator
        pathway contributions, and per-channel direct contributions."""
        posterior = self._posterior
        if posterior is None:
            return
        try:
            n_obs = int(getattr(self.model, "n_obs", 0))
            components: dict[str, np.ndarray] = {}

            intercept = self._mean("alpha_y")
            if intercept is not None and n_obs:
                # alpha_y is on the standardized scale; the baseline carries
                # the outcome's location shift
                y_mean, y_std = self._outcome_scale()
                components["Baseline"] = np.full(n_obs, intercept * y_std + y_mean)

            for med in getattr(self.model, "mediator_names", []) or []:
                effect = self._obs_mean(f"effect_{med}_on_y")
                if effect is not None:
                    components[f"Via {med}"] = effect

            for ch in self._get_channel_names():
                direct = self._obs_mean(f"direct_effect_{ch}")
                if direct is not None:
                    components[f"{ch} (direct)"] = direct

            if components:
                bundle.component_time_series = components
                bundle.component_totals = {
                    name: float(vals.sum()) for name, vals in components.items()
                }
        except Exception as e:
            logger.warning(f"Nested decomposition extraction failed: {e}")

    def _multioutcome_components(self, k: int) -> dict[str, np.ndarray] | None:
        """Component time series for outcome ``k`` of a multi-outcome model.

        Channel contributions re-evaluate the media transform at posterior-mean
        parameters; any cross-effect/mediator remainder is reported as the
        difference between ``mu`` and the explicit components so the stack sums
        to the model's prediction.
        """
        posterior = self._posterior
        if posterior is None or "mu" not in posterior:
            return None

        outcome = self.model.outcome_names[k]
        channels = self._get_channel_names()
        mu_mean = np.asarray(posterior["mu"].values, dtype=float).mean(axis=(0, 1))[
            :, k
        ]
        n_obs = len(mu_mean)

        components: dict[str, np.ndarray] = {}
        out_mean, out_std = self._outcome_scale(outcome)

        # Baseline (standardized-scale outcome intercept): "alpha" in
        # MultivariateMMM, "alpha_y" (dims outcome) in CombinedMMM. The
        # baseline carries the outcome's location shift in original units.
        intercept = None
        for name in ("alpha", "alpha_y"):
            if name in posterior:
                vals = np.asarray(posterior[name].values, dtype=float)
                if vals.ndim == 3 and vals.shape[-1] == len(self.model.outcome_names):
                    intercept = float(vals[:, :, k].mean())
                    break
        if intercept is not None:
            components["Baseline"] = np.full(n_obs, intercept * out_std + out_mean)

        # Media contributions: beta[outcome, channel] * x_sat(channel),
        # de-standardized into original units
        beta_name = (
            "beta_media"
            if "beta_media" in posterior
            else "beta_direct" if "beta_direct" in posterior else None
        )
        if beta_name is not None:
            beta = np.asarray(posterior[beta_name].values, dtype=float).mean(
                axis=(0, 1)
            )
            for c, ch in enumerate(channels):
                x_sat = self._media_x_sat(c, ch)
                if x_sat is not None:
                    components[ch] = beta[k, c] * x_sat * out_std

        # Mediator contributions (CombinedMMM): gamma[outcome, mediator] * latent
        if "gamma" in posterior:
            gamma = np.asarray(posterior["gamma"].values, dtype=float).mean(axis=(0, 1))
            for m, med in enumerate(getattr(self.model, "mediator_names", []) or []):
                affected = None
                try:
                    affected = self.model._get_affected_outcomes(med)
                except Exception:
                    pass
                if affected is not None and outcome not in affected:
                    continue
                latent = self._obs_mean(f"{med}_latent")
                if latent is not None and gamma.ndim == 2:
                    components[f"Via {med}"] = gamma[k, m] * latent * out_std

        if not components:
            return None

        # Cross-effect (and approximation) remainder, so components stack to mu
        explained = np.sum(list(components.values()), axis=0)
        remainder = mu_mean - explained
        if "psi_matrix" in posterior and np.max(np.abs(remainder)) > 1e-9:
            components["Cross-outcome effects"] = remainder

        return components

    def _extract_multioutcome_decomposition(self, bundle: MMMDataBundle) -> None:
        """Per-outcome decomposition for MultivariateMMM/CombinedMMM."""
        try:
            outcome_names = list(self.model.outcome_names)
            by_outcome: dict[str, dict[str, np.ndarray]] = {}
            for k, name in enumerate(outcome_names):
                components = self._multioutcome_components(k)
                if components:
                    by_outcome[name] = components

            if not by_outcome:
                return

            primary = outcome_names[0]
            if primary in by_outcome:
                bundle.component_time_series = by_outcome[primary]
                bundle.component_totals = {
                    name: float(vals.sum())
                    for name, vals in by_outcome[primary].items()
                }

            bundle.component_time_series_by_product = by_outcome
            bundle.component_totals_by_product = {
                outcome: {name: float(vals.sum()) for name, vals in comps.items()}
                for outcome, comps in by_outcome.items()
            }
        except Exception as e:
            logger.warning(f"Multi-outcome decomposition extraction failed: {e}")

    # ------------------------------------------------------------------
    # Cross-effects / correlations (MultivariateMMM / CombinedMMM)
    # ------------------------------------------------------------------

    def _extract_cross_effects(self, bundle: MMMDataBundle) -> None:
        """Cannibalization matrix, net effects, and outcome correlations."""
        posterior = self._posterior
        if posterior is None:
            return
        try:
            outcome_names = list(self.model.outcome_names)
            bundle.product_names = bundle.product_names or outcome_names

            # Outcome residual correlations
            for name in ("Y_obs_correlation", "corr", "corr_chol"):
                if name in posterior:
                    vals = np.asarray(posterior[name].values, dtype=float).mean(
                        axis=(0, 1)
                    )
                    if name == "corr_chol":
                        vals = vals @ vals.T
                    bundle.outcome_correlations = vals
                    break

            if "psi_matrix" not in posterior:
                return

            psi = np.asarray(posterior["psi_matrix"].values, dtype=float)
            psi_samples = psi.reshape(-1, *psi.shape[2:])  # (S, source, target)
            psi_mean = psi_samples.mean(axis=0)

            specs = getattr(self.model, "_cross_effect_specs", None) or []
            matrix: dict[str, dict[str, dict[str, float]]] = {}
            for spec in specs:
                source = outcome_names[spec.source_idx]
                target = outcome_names[spec.target_idx]
                matrix.setdefault(source, {})[target] = self._stats(
                    psi_samples[:, spec.source_idx, spec.target_idx]
                )
            if matrix:
                bundle.cannibalization_matrix = matrix

            # Net product effects: direct media contribution vs cross-outcome
            # contribution received (psi[source, target] * Y_source).
            outcome_data = getattr(self.model, "outcome_data", None)
            components_ok = bundle.component_totals_by_product is not None
            if outcome_data and components_ok:
                net: dict[str, dict[str, float]] = {}
                channels = set(self._get_channel_names())
                for t, target in enumerate(outcome_names):
                    totals = bundle.component_totals_by_product.get(target, {})
                    direct = sum(v for name, v in totals.items() if name in channels)
                    cross = 0.0
                    for s, source in enumerate(outcome_names):
                        if s == t or psi_mean[s, t] == 0:
                            continue
                        y_source = np.asarray(outcome_data[source], dtype=float)
                        cross += float(psi_mean[s, t] * y_source.sum())
                    net[target] = {
                        "direct": float(direct),
                        "cannibalization": cross,
                        "net": float(direct) + cross,
                    }
                bundle.net_product_effects = net
        except Exception as e:
            logger.warning(f"Cross-effect extraction failed: {e}")

    # ------------------------------------------------------------------
    # ROI, saturation, adstock, spend
    # ------------------------------------------------------------------

    def _compute_channel_roi(self) -> dict[str, dict[str, float]] | None:
        """Channel ROI on the (primary) outcome.

        Contribution uses coefficient posterior samples applied to the
        posterior-mean saturated spend; spend is the raw channel total.
        """
        try:
            channels = self._get_channel_names()
            X = np.asarray(self.model.X_media, dtype=float)
            roi: dict[str, dict[str, float]] = {}
            for c, ch in enumerate(channels):
                spend = float(X[:, c].sum())
                if spend <= 0:
                    continue
                coef = self._total_effect_samples(ch)
                x_sat = self._media_x_sat(c, ch)
                if coef is None or x_sat is None:
                    continue
                contribution = coef * float(x_sat.sum())
                roi[ch] = self._stats(contribution / spend)
            return roi or None
        except Exception as e:
            logger.warning(f"Extended channel ROI failed: {e}")
            return None

    def _get_saturation_curves(self) -> dict[str, dict[str, np.ndarray]] | None:
        """Logistic saturation response curves at posterior-mean parameters.

        The x-axis is weekly spend; saturation is evaluated at the model's own
        normalization (spend / per-channel raw max), and the response is the
        channel's contribution in original KPI units.
        """
        try:
            channels = self._get_channel_names()
            X = np.asarray(self.model.X_media, dtype=float)
            scales = getattr(self.model, "_media_scale", None)
            if scales is None:
                scales = X.max(axis=0) + 1e-8
            curves: dict[str, dict[str, np.ndarray]] = {}
            for c, ch in enumerate(channels):
                lam = self._mean(f"lambda_{ch}")
                if lam is None:
                    continue
                media_scale = float(scales[c])
                spend = np.linspace(0, 2.0 * media_scale, 100)
                response = 1 - np.exp(-lam * spend / media_scale)
                coef = self._total_effect_samples(ch)
                if coef is not None:
                    response = response * float(np.mean(coef))
                curves[ch] = {"spend": spend, "response": response}
            return curves or None
        except Exception as e:
            logger.warning(f"Extended saturation curves failed: {e}")
            return None

    def _get_adstock_curves(self) -> dict[str, np.ndarray] | None:
        """Geometric adstock kernels at posterior-mean decay."""
        try:
            curves: dict[str, np.ndarray] = {}
            for ch in self._get_channel_names():
                alpha = self._mean(f"alpha_{ch}")
                if alpha is None:
                    alpha = self._mean("alpha_shared")
                if alpha is None or not 0 < alpha < 1:
                    continue
                curves[ch] = adstock_weights("geometric", _ADSTOCK_L_MAX, alpha=alpha)
            return curves or None
        except Exception as e:
            logger.warning(f"Extended adstock curves failed: {e}")
            return None

    def _extract_summary_metrics(self, bundle: MMMDataBundle) -> None:
        """Executive-summary metrics on the primary outcome: total
        marketing-attributed revenue, blended ROI, and contribution share."""
        try:
            channels = self._get_channel_names()
            X = np.asarray(self.model.X_media, dtype=float)
            contrib_samples = None
            total_spend = 0.0
            for c, ch in enumerate(channels):
                coef = self._total_effect_samples(ch)
                x_sat = self._media_x_sat(c, ch)
                if coef is None or x_sat is None:
                    continue
                contribution = coef * float(x_sat.sum())
                contrib_samples = (
                    contribution
                    if contrib_samples is None
                    else contrib_samples + contribution
                )
                total_spend += float(X[:, c].sum())

            if contrib_samples is None:
                return

            bundle.marketing_attributed_revenue = self._stats(contrib_samples)
            if total_spend > 0:
                bundle.blended_roi = self._stats(contrib_samples / total_spend)
            if bundle.total_revenue:
                bundle.marketing_contribution_pct = self._stats(
                    contrib_samples / bundle.total_revenue
                )
        except Exception as e:
            logger.warning(f"Extended summary metrics failed: {e}")

    def _get_current_spend(self) -> dict[str, float] | None:
        """Typical (mean) weekly spend per channel — the saturation curves'
        x-axis is weekly spend, so the current-spend markers must match."""
        try:
            X = np.asarray(self.model.X_media, dtype=float)
            return {
                ch: float(X[:, c].mean())
                for c, ch in enumerate(self._get_channel_names())
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Specification
    # ------------------------------------------------------------------

    def _get_model_specification(self) -> dict[str, Any]:
        """Get model specification for extended models."""
        spec: dict[str, Any] = {
            "model_type": type(self.model).__name__,
            "likelihood": "Normal",
            "adstock": "geometric (estimated decay)",
            "saturation": "logistic",
        }

        if getattr(self.model, "mediator_names", None):
            spec["mediators"] = list(self.model.mediator_names)
        if getattr(self.model, "outcome_names", None):
            spec["outcomes"] = list(self.model.outcome_names)
            spec["likelihood"] = "Multivariate normal (LKJ residual correlation)"
        if getattr(self.model, "experiments", None):
            spec["experiment_calibrations"] = len(self.model.experiments)

        return spec


__all__ = ["ExtendedMMMExtractor"]
