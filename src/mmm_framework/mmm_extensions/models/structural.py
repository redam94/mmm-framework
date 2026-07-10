"""
StructuralNestedMMM -- a multi-mediator structural MMM.

Generalizes NestedMMM's single-shape mediator block into a structural
equation system over a DAG of latent states, each with its own dynamics
(static / AR(1) / random walk) and measurement model (standardized Gaussian,
binomial survey with weekly-varying sample size, cumulative-logit Likert,
or fully latent), plus shared latent factors (e.g. a demand trend) entering
mediator equations and the outcome.

Motivating example::

    TV -> awareness            (binary tracker, n_t respondents/week, AR(1)
                                persistence: population awareness carries over)
    Display + Social + awareness + price + demand -> consideration
                               (5-point Likert survey)
    consideration + demand + Search -> sales

Design spec: technical-docs/structural-nested-mmm.md.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .base import BaseExtendedMMM
from ..results import MediationEffects

if TYPE_CHECKING:
    from ..config import LatentFactorSpec, MediatorSpec, StructuralNestedConfig

_MIN_OBSERVED_POINTS = 3
_SOFT_OBSERVED_POINTS = 8


class StructuralNestedMMM(BaseExtendedMMM):
    """MMM with a DAG of structural mediator equations.

    Parameters
    ----------
    X_media : np.ndarray
        Raw media matrix (n_obs, n_channels).
    y : np.ndarray
        Outcome (n_obs,), raw scale (standardized internally).
    channel_names : list[str]
        Media channel names.
    config : StructuralNestedConfig
        The mediator DAG + latent factors + outcome settings.
    mediator_data : dict[str, np.ndarray] | None
        Per-mediator observations. GAUSSIAN: (n_obs,) float, NaN = unobserved.
        BINOMIAL: (n_obs,) success counts, NaN = no survey that week.
        ORDERED: (n_obs, K) per-category counts, NaN/all-zero row = unobserved.
        LATENT mediators must not appear here.
    mediator_trials : dict[str, np.ndarray] | None
        BINOMIAL only: (n_obs,) respondents per week (the weekly survey
        volume); NaN/0 = unobserved week.
    X_controls : np.ndarray | None
        Control matrix (n_obs, n_controls); z-scored internally.
    control_names : list[str] | None
        Control column names (required with X_controls).
    index, model_config, trend_config
        As in :class:`BaseExtendedMMM` (trend/seasonality/likelihood reuse the
        shared extension components).
    """

    def __init__(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        config: "StructuralNestedConfig",
        mediator_data: dict[str, np.ndarray] | None = None,
        mediator_trials: dict[str, np.ndarray] | None = None,
        X_controls: np.ndarray | None = None,
        control_names: list[str] | None = None,
        index: pd.Index | None = None,
        model_config=None,
        trend_config=None,
    ):
        super().__init__(
            X_media,
            y,
            channel_names,
            index,
            model_config=model_config,
            trend_config=trend_config,
        )
        self.config = config
        self.mediator_data = {
            k: np.asarray(v, dtype=float) for k, v in (mediator_data or {}).items()
        }
        self.mediator_trials = {
            k: np.asarray(v, dtype=float) for k, v in (mediator_trials or {}).items()
        }

        self.control_names = list(control_names or [])
        if X_controls is not None:
            if not self.control_names:
                raise ValueError("control_names is required when X_controls is given")
            Xc = np.asarray(X_controls, dtype=float)
            if Xc.ndim != 2 or Xc.shape != (self.n_obs, len(self.control_names)):
                raise ValueError(
                    f"X_controls must be (n_obs={self.n_obs}, "
                    f"n_controls={len(self.control_names)}); got {Xc.shape}"
                )
            # z-score: mediator/outcome equations use inline Normal(0, 1) betas
            # on standardized controls (the recovery-search lesson).
            self._controls_mean = Xc.mean(axis=0)
            self._controls_std = Xc.std(axis=0) + 1e-8
            self.X_controls = (Xc - self._controls_mean) / self._controls_std
        else:
            if self.control_names:
                raise ValueError("X_controls is required when control_names is given")
            self.X_controls = None

        self.mediator_names = [m.name for m in config.mediators]
        self.n_mediators = len(config.mediators)
        self._med_by_name = {m.name: m for m in config.mediators}
        self._factor_by_name = {f.name: f for f in config.latent_factors}
        self._topo_order = config.topological_order()

        self.mediator_masks: dict[str, np.ndarray] = {}
        self._gaussian_std: dict[str, tuple[np.ndarray, float, float]] = {}
        self._validate_inputs()

    # ------------------------------------------------------------------
    # Validation / data prep
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        from ..config import MediatorLikelihood

        control_set = set(self.control_names)
        reserved = set(self.channel_names) | control_set
        for nm in [m.name for m in self.config.mediators] + [
            f.name for f in self.config.latent_factors
        ]:
            if nm in reserved:
                raise ValueError(
                    f"Mediator/factor name '{nm}' collides with a channel or "
                    "control column name -- RV names would clash"
                )
        for spec in self.config.mediators:
            unknown_ch = set(spec.channels) - set(self.channel_names)
            if unknown_ch:
                raise ValueError(
                    f"Mediator '{spec.name}' references unknown channels: "
                    f"{sorted(unknown_ch)}"
                )
            unknown_cv = set(spec.controls) - control_set
            if unknown_cv:
                raise ValueError(
                    f"Mediator '{spec.name}' references unknown controls: "
                    f"{sorted(unknown_cv)}"
                )
        if self.config.outcome_controls is not None:
            unknown = set(self.config.outcome_controls) - control_set
            if unknown:
                raise ValueError(
                    f"outcome_controls not in control_names: {sorted(unknown)}"
                )

        med_names = {m.name for m in self.config.mediators}
        stray_data = set(self.mediator_data) - med_names
        if stray_data:
            raise ValueError(
                f"mediator_data keys name no configured mediator: {sorted(stray_data)}"
            )
        from ..config import MediatorLikelihood as _ML

        binomial_names = {
            m.name
            for m in self.config.mediators
            if m.measurement.likelihood == _ML.BINOMIAL
        }
        stray_trials = set(self.mediator_trials) - binomial_names
        if stray_trials:
            raise ValueError(
                "mediator_trials keys name no BINOMIAL mediator: "
                f"{sorted(stray_trials)}"
            )

        for spec in self.config.mediators:
            lk = spec.measurement.likelihood
            name = spec.name

            if lk == MediatorLikelihood.LATENT:
                if name in self.mediator_data:
                    raise ValueError(
                        f"Mediator '{name}' is configured LATENT but has data -- "
                        "set a measurement likelihood or drop the data"
                    )
                continue

            if name not in self.mediator_data:
                raise ValueError(
                    f"Mediator '{name}' ({lk.value}) requires mediator_data[{name!r}]"
                )
            data = self.mediator_data[name]

            if lk == MediatorLikelihood.GAUSSIAN:
                if data.shape != (self.n_obs,):
                    raise ValueError(
                        f"Gaussian mediator '{name}' data must be (n_obs,); "
                        f"got {data.shape}"
                    )
                mask = np.isfinite(data)
                self._check_min_obs(name, mask)
                obs = data[mask]
                mean, std = float(obs.mean()), float(obs.std()) + 1e-8
                std_data = np.full(self.n_obs, np.nan)
                std_data[mask] = (obs - mean) / std
                self._gaussian_std[name] = (std_data, mean, std)
                self.mediator_masks[name] = mask

            elif lk == MediatorLikelihood.BINOMIAL:
                if data.shape != (self.n_obs,):
                    raise ValueError(
                        f"Binomial mediator '{name}' counts must be (n_obs,); "
                        f"got {data.shape}"
                    )
                if name not in self.mediator_trials:
                    raise ValueError(
                        f"Binomial mediator '{name}' requires mediator_trials[{name!r}] "
                        "(per-period survey sample sizes)"
                    )
                trials = self.mediator_trials[name]
                if trials.shape != (self.n_obs,):
                    raise ValueError(
                        f"mediator_trials[{name!r}] must be (n_obs,); got {trials.shape}"
                    )
                if np.any(np.nan_to_num(data) < 0):
                    raise ValueError(f"Binomial mediator '{name}': negative counts")
                mask = np.isfinite(data) & np.isfinite(trials) & (trials > 0)
                self._check_min_obs(name, mask)
                if np.any(data[mask] > trials[mask]):
                    raise ValueError(
                        f"Binomial mediator '{name}': counts exceed trials on "
                        "observed weeks"
                    )
                self.mediator_masks[name] = mask

            elif lk == MediatorLikelihood.ORDERED:
                K = spec.measurement.n_categories
                if data.ndim != 2 or data.shape != (self.n_obs, K):
                    raise ValueError(
                        f"Ordered mediator '{name}' data must be (n_obs, "
                        f"K={K}) category counts; got {data.shape}"
                    )
                row_ok = np.isfinite(data).all(axis=1)
                row_sum = np.where(row_ok, np.nan_to_num(data).sum(axis=1), 0.0)
                mask = row_ok & (row_sum > 0)
                if np.any(np.nan_to_num(data) < 0):
                    raise ValueError(f"Ordered mediator '{name}': negative counts")
                partial = int((np.isfinite(data).any(axis=1) & ~row_ok).sum())
                if partial:
                    warnings.warn(
                        f"Ordered mediator '{name}': {partial} week(s) have "
                        "counts for SOME categories only -- treated as "
                        "unobserved (a survey week needs all K category counts).",
                        stacklevel=4,
                    )
                self._check_min_obs(name, mask)
                self.mediator_masks[name] = mask

    def _check_min_obs(self, name: str, mask: np.ndarray) -> None:
        n = int(mask.sum())
        if n < _MIN_OBSERVED_POINTS:
            raise ValueError(
                f"Mediator '{name}' has {n} observed points; "
                f"at least {_MIN_OBSERVED_POINTS} are required"
            )
        if n < _SOFT_OBSERVED_POINTS:
            warnings.warn(
                f"Mediator '{name}' has only {n} observed points -- the "
                "media->mediator path will be weakly identified.",
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # Coords / helpers
    # ------------------------------------------------------------------

    def _build_coords(self) -> dict:
        coords = super()._build_coords()
        coords["mediator"] = self.mediator_names
        if self.control_names:
            coords["control"] = self.control_names
        return coords

    def _get_affecting_channels(self, mediator_name: str) -> list[str]:
        """Channels entering a mediator's equation (extractor contract)."""
        return list(self._med_by_name[mediator_name].channels)

    def _channel_mediated_exact(self, channel: str) -> bool:
        """True when every mediator reachable from ``channel`` is STATIC and
        non-BINOMIAL, so the linear coefficient product is the exact
        derivative (experiment-calibration eligibility)."""
        from ..config import MediatorDynamics, MediatorLikelihood

        start = [m.name for m in self.config.mediators if channel in m.channels]
        if not start:
            return True  # direct-only channel
        reach = set(start)
        frontier = list(start)
        while frontier:
            node = frontier.pop()
            for m in self.config.mediators:
                if node in m.parents and m.name not in reach:
                    reach.add(m.name)
                    frontier.append(m.name)
        for name in reach:
            spec = self._med_by_name[name]
            if spec.dynamics != MediatorDynamics.STATIC:
                return False
            if spec.measurement.likelihood == MediatorLikelihood.BINOMIAL:
                return False
            if channel in spec.channels and not spec.adstock_enabled:
                # The handle's x_sat regressor is the ADSTOCKED series; a
                # saturation-only entry would mismatch the regressor.
                return False
        return True

    def _factor_anchor_consumer(self, factor: "LatentFactorSpec") -> str:
        """Where the factor's sign is pinned (that loading is HalfNormal).

        Explicit ``factor.anchor`` wins. "auto" prefers the first MEASURED
        mediator consumer (topological order): the anchor must be a loading
        the data can hold materially nonzero, or the reflected factor mode
        escapes the HalfNormal by sitting at its cost-free zero mode (seen as
        split-chain R-hat ~1.75 in the brand-funnel recovery when the anchor
        was the small outcome loading). Falls back to the outcome, then to any
        consumer."""
        from ..config import MediatorLikelihood

        if factor.anchor != "auto":
            return factor.anchor
        for name in self._topo_order:
            spec = self._med_by_name[name]
            if (
                factor.name in spec.latent_factors
                and spec.measurement.likelihood != MediatorLikelihood.LATENT
            ):
                return name
        if factor.affects_outcome:
            return "outcome"
        for name in self._topo_order:
            if factor.name in self._med_by_name[name].latent_factors:
                return name
        # Config validation guarantees a consumer exists.
        raise RuntimeError(f"factor '{factor.name}' has no consumer")

    def _instant_saturation(
        self,
        x_media: "pt.TensorVariable",
        channel_idx: int,
        lam: "pt.TensorVariable",
    ) -> "pt.TensorVariable":
        """Saturation-only media transform (no adstock) sharing the channel's
        ``lambda`` RV -- used when a mediator's AR(1) state supplies the
        carryover (``apply_adstock=False``)."""
        from ..components.transforms import logistic_saturation_pt

        scale = float(self._media_scale[channel_idx])
        return logistic_saturation_pt(x_media[:, channel_idx] / scale, lam)

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _build_model(self) -> pm.Model:
        from ..components.latent_states import (
            ar1_decay_matrix,
            build_binomial_state_measurement,
            build_gaussian_state_measurement,
            build_latent_state,
            build_ordered_state_measurement,
            standardize_in_graph,
        )

        # noqa hint: standardize_in_graph is used for latent FACTORS only
        # (media-independent, so counterfactual-safe -- see latent_states.py).
        from ..components.outcome import build_outcome_likelihood
        from ..components.priors import create_effect_prior
        from ..components.temporal import (
            build_seasonality_contribution,
            build_trend_contribution,
        )
        from ..config import MediatorDynamics, MediatorLikelihood

        coords = self._build_coords()
        n_obs = self.n_obs

        # Graph handles captured locally for experiment calibration + the
        # linearized pathway deterministics (never stored on self -- graph
        # objects don't survive pickling).
        channel_tx: dict = {}
        med_betas: dict[str, dict[str, pt.TensorVariable]] = {}
        parent_lambdas: dict[tuple[str, str], pt.TensorVariable] = {}
        gammas: dict[str, pt.TensorVariable] = {}
        deltas: dict[str, pt.TensorVariable] = {}
        med_gain: dict[str, pt.TensorVariable] = {}

        with pm.Model(coords=coords) as model:
            X_media = pm.Data("X_media", self.X_media, dims=("obs", "channel"))
            if self.X_controls is not None:
                X_controls = pm.Data(
                    "X_controls", self.X_controls, dims=("obs", "control")
                )
            y_standardized = (
                np.asarray(self.y, dtype=float) - self.y_mean
            ) / self.y_std
            y = pm.Data("y", y_standardized, dims="obs")

            # ---- media transforms --------------------------------------
            # One adstocked + saturated series per channel where consumed,
            # plus a saturation-only variant for equations whose AR(1) state
            # carries the media effect forward itself. The adstock alpha RV is
            # created ONLY when some equation actually consumes the adstocked
            # series (an adstock-enabled mediator, or the channel's own direct
            # outcome path) -- otherwise it would be a dead free RV sampling
            # its prior.
            needs_instant = {
                ch
                for spec in self.config.mediators
                if not spec.adstock_enabled
                for ch in spec.channels
            }
            mediated_channels = {
                ch for spec in self.config.mediators for ch in spec.channels
            }

            def _has_direct_outcome_path(ch: str) -> bool:
                if ch not in mediated_channels:
                    return True  # plain beta_<ch> path
                return any(
                    m.allow_direct_effect and ch in m.channels
                    for m in self.config.mediators
                )

            needs_adstocked = {
                ch
                for i, ch in enumerate(self.channel_names)
                if _has_direct_outcome_path(ch)
                or any(
                    ch in spec.channels and spec.adstock_enabled
                    for spec in self.config.mediators
                )
            }
            sat_cols: list[pt.TensorVariable] = []
            sat_instant: dict[str, pt.TensorVariable] = {}
            for i, channel in enumerate(self.channel_names):
                lam = pm.Gamma(f"lambda_{channel}", alpha=3, beta=1)
                if channel in needs_instant:
                    sat_instant[channel] = self._instant_saturation(X_media, i, lam)
                x = X_media[:, i]
                if channel in needs_adstocked:
                    alpha = pm.Beta(f"alpha_{channel}", alpha=2, beta=2)
                    apply = self._media_transform_apply(i, alpha, lam)
                    x_sat = apply(x)
                else:
                    # Only saturation-only consumers exist; keep the instant
                    # series in the stacked matrix so shapes stay uniform.
                    apply = None
                    x_sat = sat_instant[channel]
                sat_cols.append(x_sat)
                channel_tx[channel] = (x_sat, apply, x)
            sat = pt.stack(sat_cols, axis=1)

            # ---- latent factors ----------------------------------------
            factor_signals: dict[str, pt.TensorVariable] = {}
            for f in self.config.latent_factors:
                # Small deterministic nonzero initval: at an all-zero start the
                # standardized factor sits on a saddle (F identically 0, no
                # gradient into the loadings) and MAP/ADVI stall there.
                eps = pm.Normal(
                    f"{f.name}_innovation",
                    mu=0.0,
                    sigma=1.0,
                    shape=n_obs,
                    initval=np.linspace(-0.1, 0.1, n_obs),
                )
                if f.dynamics == MediatorDynamics.AR1:
                    rho = pm.Beta(
                        f"{f.name}_persistence",
                        alpha=f.rho_prior_alpha,
                        beta=f.rho_prior_beta,
                    )
                    raw = ar1_decay_matrix(pt.clip(rho, 1e-6, 1 - 1e-6), n_obs) @ eps
                elif f.dynamics == MediatorDynamics.RANDOM_WALK:
                    raw = pt.cumsum(eps)
                else:  # STATIC: iid factor (CFA-style)
                    raw = eps
                F = standardize_in_graph(raw)
                pm.Deterministic(f"factor_{f.name}", F, dims="obs")
                factor_signals[f.name] = F

            # ---- mediator equations (topological order) ----------------
            med_signals: dict[str, pt.TensorVariable] = {}
            for name in self._topo_order:
                spec = self._med_by_name[name]
                lk = spec.measurement.likelihood
                drivers = pt.zeros(n_obs)

                betas_here: dict[str, pt.TensorVariable] = {}
                for ch in spec.channels:
                    i = self.channel_names.index(ch)
                    x_in = sat_instant[ch] if not spec.adstock_enabled else sat[:, i]
                    b = create_effect_prior(
                        f"beta_{ch}_to_{name}",
                        constrained=spec.media_effect.constraint.value,
                        mu=spec.media_effect.mu,
                        sigma=spec.media_effect.sigma,
                    )
                    drivers = drivers + b * x_in
                    betas_here[ch] = b
                med_betas[name] = betas_here

                for p in spec.parents:
                    lam_p = create_effect_prior(
                        f"lambda_{p}_to_{name}",
                        constrained=spec.parent_effect.constraint.value,
                        mu=spec.parent_effect.mu,
                        sigma=spec.parent_effect.sigma,
                    )
                    drivers = drivers + lam_p * med_signals[p]
                    parent_lambdas[(p, name)] = lam_p

                for cv in spec.controls:
                    k = self.control_names.index(cv)
                    phi = create_effect_prior(
                        f"phi_{cv}_to_{name}",
                        constrained=spec.control_effect.constraint.value,
                        mu=spec.control_effect.mu,
                        sigma=spec.control_effect.sigma,
                    )
                    drivers = drivers + phi * X_controls[:, k]

                for fn in spec.latent_factors:
                    f = self._factor_by_name[fn]
                    if self._factor_anchor_consumer(f) == name:
                        w = pm.HalfNormal(
                            f"w_{fn}_to_{name}", sigma=f.mediator_effect_sigma
                        )
                    else:
                        w = pm.Normal(
                            f"w_{fn}_to_{name}", mu=0.0, sigma=f.mediator_effect_sigma
                        )
                    drivers = drivers + w * factor_signals[fn]

                # Level: ordered measurements have no free level (cutpoints
                # absorb location -- exactly confounded otherwise); RANDOM_WALK
                # states have none either (level <-> first innovation is an
                # exact ridge -- the walk's start carries location).
                if (
                    lk == MediatorLikelihood.ORDERED
                    or spec.dynamics == MediatorDynamics.RANDOM_WALK
                ):
                    level: pt.TensorVariable | float = 0.0
                else:
                    level = pm.Normal(f"level_{name}", mu=0.0, sigma=2.0)

                # Process noise needs a measurement to be identified (an
                # unmeasured state's innovations would just absorb outcome
                # residual). Measured non-Gaussian STATIC states keep an iid
                # noise term as overdispersion slack: a large-n_t tracker's
                # binomial variance is far tighter than real week-to-week
                # methodology wobble, and the misfit needs somewhere to go.
                measured = lk != MediatorLikelihood.LATENT
                non_gaussian = lk in (
                    MediatorLikelihood.BINOMIAL,
                    MediatorLikelihood.ORDERED,
                )
                innovation = (
                    spec.innovation_sigma
                    if (
                        measured
                        and (spec.dynamics != MediatorDynamics.STATIC or non_gaussian)
                    )
                    else None
                )
                if (
                    spec.adstock_enabled
                    and spec.channels
                    and spec.dynamics != MediatorDynamics.STATIC
                ):
                    warnings.warn(
                        f"Mediator '{name}' has {spec.dynamics.value} dynamics AND "
                        "adstocked media inputs -- two nearly-interchangeable "
                        "geometric carryovers (alpha vs rho ridge). Prefer "
                        "apply_adstock=False so the state carries the memory.",
                        stacklevel=2,
                    )
                # "auto" state parameterization: a densely observed tracker
                # (>= 50% of weeks) pins z_t hard enough that non-centered
                # innovations funnel against sigma -- sample the AR noise
                # directly (centered) there; sparse measurements keep the
                # non-centered form.
                if spec.state_parameterization == "auto":
                    frac = float(self.mediator_masks[name].mean()) if measured else 0.0
                    centered = measured and frac >= 0.5
                else:
                    centered = spec.state_parameterization == "centered"

                z = build_latent_state(
                    name,
                    drivers,
                    spec.dynamics,
                    n_obs,
                    level=level,
                    rho_prior_alpha=spec.rho_prior_alpha,
                    rho_prior_beta=spec.rho_prior_beta,
                    innovation_sigma=innovation,
                    centered=centered,
                )
                # LATENT mediators are deliberately NOT standardized in-graph:
                # their constants would be media-dependent, so a counterfactual
                # set_data swap would recompute them and contaminate every
                # contrast (and a common rescale of the driver betas would be
                # an exactly flat direction). The beta*gamma path products are
                # identified through the priors; individual edges are not --
                # documented in the spec.
                z = pm.Deterministic(f"{name}_latent", z, dims="obs")

                # AR steady-state gain (for the linearized pathway table).
                if spec.dynamics == MediatorDynamics.AR1:
                    med_gain[name] = 1.0 / (
                        1.0 - pt.clip(model[f"{name}_persistence"], 1e-6, 1 - 1e-6)
                    )
                elif spec.dynamics == MediatorDynamics.RANDOM_WALK:
                    # Average accumulation of a sustained impulse over the
                    # window -- a documented approximation for the RW case.
                    med_gain[name] = pt.as_tensor_variable((n_obs + 1) / 2.0)
                else:
                    med_gain[name] = pt.as_tensor_variable(1.0)

                # Measurement + downstream signal (natural scale, §4.5).
                if lk == MediatorLikelihood.GAUSSIAN:
                    std_data, _, _ = self._gaussian_std[name]
                    build_gaussian_state_measurement(
                        name,
                        z,
                        std_data,
                        self.mediator_masks[name],
                        noise_sigma=spec.measurement.noise_sigma,
                    )
                    signal = z
                elif lk == MediatorLikelihood.BINOMIAL:
                    p_full = build_binomial_state_measurement(
                        name,
                        z,
                        self.mediator_data[name],
                        self.mediator_trials[name],
                        self.mediator_masks[name],
                        design_effect=spec.measurement.design_effect,
                    )
                    pm.Deterministic(f"{name}_probability", p_full, dims="obs")
                    signal = p_full
                elif lk == MediatorLikelihood.ORDERED:
                    build_ordered_state_measurement(
                        name,
                        z,
                        self.mediator_data[name],
                        self.mediator_masks[name],
                        n_categories=spec.measurement.n_categories,
                        cutpoint_prior_sigma=spec.measurement.cutpoint_prior_sigma,
                        design_effect=spec.measurement.design_effect,
                    )
                    signal = z
                else:
                    signal = z

                # Downstream consumers read the UNCENTERED natural-scale
                # signal. In-graph centering would be recomputed under a
                # counterfactual set_data swap, making the summed mediated
                # contrast identically zero (design-review blocker); the
                # consumer's intercept absorbs the signal's level instead,
                # exactly as in NestedMMM.
                med_signals[name] = signal

            # ---- outcome equation --------------------------------------
            alpha_y = pm.Normal("alpha_y", mu=0.0, sigma=2.0)
            mu_std = alpha_y + pt.zeros(n_obs)

            for spec in self.config.mediators:
                if not spec.affects_outcome:
                    continue
                gamma = create_effect_prior(
                    f"gamma_{spec.name}",
                    constrained=spec.outcome_effect.constraint.value,
                    mu=spec.outcome_effect.mu,
                    sigma=spec.outcome_effect.sigma,
                )
                gammas[spec.name] = gamma
                contrib = gamma * med_signals[spec.name]
                mu_std = mu_std + contrib
                pm.Deterministic(
                    f"effect_{spec.name}_on_y", contrib * self.y_std, dims="obs"
                )

            # Direct media effects. Mediated channels get a (tight) delta when
            # a routing mediator allows it; unrouted channels get a plain beta.
            mediated_channels = {
                ch for spec in self.config.mediators for ch in spec.channels
            }
            for i, channel in enumerate(self.channel_names):
                if channel in mediated_channels:
                    granting = next(
                        (
                            m
                            for m in self.config.mediators
                            if m.allow_direct_effect and channel in m.channels
                        ),
                        None,
                    )
                    if granting is None:
                        continue
                    coef = create_effect_prior(
                        f"delta_direct_{channel}",
                        constrained=granting.direct_effect.constraint.value,
                        mu=granting.direct_effect.mu,
                        sigma=granting.direct_effect.sigma,
                    )
                    deltas[channel] = coef
                else:
                    coef = create_effect_prior(
                        f"beta_{channel}",
                        constrained=self.config.nonmediated_effect.constraint.value,
                        mu=self.config.nonmediated_effect.mu,
                        sigma=self.config.nonmediated_effect.sigma,
                    )
                    deltas[channel] = coef
                contrib = coef * sat[:, i]
                mu_std = mu_std + contrib
                pm.Deterministic(
                    f"direct_effect_{channel}", contrib * self.y_std, dims="obs"
                )

            # Outcome controls (inline Normal(0, 1) on standardized values).
            outcome_controls = (
                list(self.config.outcome_controls)
                if self.config.outcome_controls is not None
                else list(self.control_names)
            )
            if outcome_controls:
                ctrl_total = pt.zeros(n_obs)
                for cv in outcome_controls:
                    k = self.control_names.index(cv)
                    b = pm.Normal(f"beta_ctrl_{cv}", mu=0.0, sigma=1.0)
                    ctrl_total = ctrl_total + b * X_controls[:, k]
                mu_std = mu_std + ctrl_total
                pm.Deterministic("controls_total", ctrl_total * self.y_std, dims="obs")

            # Latent factors -> outcome. HalfNormal ONLY when the outcome is
            # the factor's sign anchor; otherwise free-sign (the sign is
            # already pinned at a measured mediator's loading).
            for f in self.config.latent_factors:
                if not f.affects_outcome:
                    continue
                if self._factor_anchor_consumer(f) == "outcome":
                    w = pm.HalfNormal(f"w_{f.name}_to_y", sigma=f.outcome_effect_sigma)
                else:
                    w = pm.Normal(
                        f"w_{f.name}_to_y", mu=0.0, sigma=f.outcome_effect_sigma
                    )
                contrib = w * factor_signals[f.name]
                mu_std = mu_std + contrib
                pm.Deterministic(
                    f"effect_factor_{f.name}_on_y", contrib * self.y_std, dims="obs"
                )

            # Baseline dynamics + likelihood (shared extension components).
            trend_contrib = build_trend_contribution("", n_obs, self.trend_config)
            if trend_contrib is not None:
                mu_std = mu_std + trend_contrib
                pm.Deterministic(
                    "trend_component", trend_contrib * self.y_std, dims="obs"
                )
            seasonality_contrib = build_seasonality_contribution(
                "",
                self.index,
                n_obs,
                getattr(self.model_config, "seasonality", None),
            )
            if seasonality_contrib is not None:
                mu_std = mu_std + seasonality_contrib
                pm.Deterministic(
                    "seasonality_component",
                    seasonality_contrib * self.y_std,
                    dims="obs",
                )

            pm.Deterministic("mu", mu_std * self.y_std + self.y_mean, dims="obs")
            build_outcome_likelihood(
                "y_obs",
                mu_std,
                y,
                getattr(self.model_config, "likelihood", None),
                dims="obs",
            )

            # Linearized per-path deterministics (report + pathway table).
            path_coefs = self._register_pathway_deterministics(
                model, med_betas, parent_lambdas, gammas, med_gain, med_signals
            )

            # Experiment calibration. Handles are attached ONLY for channels
            # whose mediated paths are exact-linear (every node STATIC and
            # non-BINOMIAL, so the coefficient product is the true derivative);
            # a channel routed through AR dynamics or a sigmoid link is skipped
            # with a warning rather than calibrated against a mis-scaled
            # estimand. Exact counterfactual effects remain available post-fit
            # via get_mediation_effects().
            if self.experiments:
                handles = {}
                inexact: list[str] = []
                for channel in self.channel_names:
                    if not self._channel_mediated_exact(channel):
                        inexact.append(channel)
                        continue
                    coef = deltas.get(channel)
                    lin = path_coefs.get(channel)
                    if lin is not None:
                        coef = lin if coef is None else coef + lin
                    if coef is None:
                        continue
                    x_sat_ch, apply, x_input = channel_tx[channel]
                    ch_idx = self.channel_names.index(channel)
                    handles[channel] = {
                        "coef": coef,
                        "x_sat": x_sat_ch,
                        "apply": apply,
                        "x_input": x_input,
                        "spend_obs": self.X_media[:, ch_idx],
                    }
                targeted = {e.channel for e in self.experiments}
                bad = sorted(targeted & set(inexact))
                if bad:
                    warnings.warn(
                        f"Experiment calibration skipped for {bad}: their "
                        "mediated paths cross AR-dynamic or binomial-link "
                        "mediators, where a linear coefficient handle would "
                        "attach the likelihood to a mis-scaled estimand.",
                        stacklevel=2,
                    )
                self._add_experiment_likelihoods(handles, scale=self.y_std)

        return model

    def _register_pathway_deterministics(
        self,
        model: pm.Model,
        med_betas: dict,
        parent_lambdas: dict,
        gammas: dict,
        med_gain: dict,
        med_signals: dict,
    ) -> dict:
        """Register ``indirect_<ch>_via_<med>`` (first-order path strength in
        original KPI units per unit saturated media) and return per-channel
        linearized total mediated coefficients (standardized scale).

        Path strength for channel c via mediator m = sum over DAG paths
        c -> m1 -> ... -> m of  beta_{c->m1} * prod(edge lambdas) *
        prod(node gains) * prod(sigmoid slopes) * gamma_m.  Node gain is the
        AR steady-state amplification 1/(1-rho); the sigmoid slope for a
        BINOMIAL node is the average p(1-p) over the window (its downstream
        signal is the probability, not the logit state). Linearized -- the
        exact channel totals come from counterfactuals (get_mediation_effects).
        """
        from ..config import MediatorLikelihood

        # Downstream derivative of each mediator's *signal* wrt its state.
        slope: dict[str, pt.TensorVariable] = {}
        for spec in self.config.mediators:
            if spec.measurement.likelihood == MediatorLikelihood.BINOMIAL:
                p = model[f"{spec.name}_probability"]
                slope[spec.name] = (p * (1.0 - p)).mean()
            else:
                slope[spec.name] = pt.as_tensor_variable(1.0)

        # d(signal_m) / d(driver injected into mediator m)
        # = gain_m * slope_m ; edges m -> child multiply by lambda.
        # Accumulate per-channel response at every mediator by walking the
        # topological order.
        per_channel_at: dict[str, dict[str, pt.TensorVariable]] = {
            ch: {} for ch in self.channel_names
        }
        for name in self._topo_order:
            spec = self._med_by_name[name]
            for ch, b in med_betas.get(name, {}).items():
                inject = b * med_gain[name] * slope[name]
                cur = per_channel_at[ch].get(name)
                per_channel_at[ch][name] = inject if cur is None else cur + inject
            for p in spec.parents:
                lam = parent_lambdas[(p, name)]
                for ch, upstream in list(per_channel_at.items()):
                    if p in upstream:
                        carried = upstream[p] * lam * med_gain[name] * slope[name]
                        cur = upstream.get(name)
                        upstream[name] = carried if cur is None else cur + carried

        path_coefs: dict[str, pt.TensorVariable] = {}
        for ch in self.channel_names:
            total = None
            for med, strength in per_channel_at[ch].items():
                if med not in gammas:
                    continue
                indirect = strength * gammas[med]
                pm.Deterministic(f"indirect_{ch}_via_{med}", indirect * self.y_std)
                total = indirect if total is None else total + indirect
            if total is not None:
                path_coefs[ch] = total
        return path_coefs

    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        nuts_sampler: str = "pymc",
        method="nuts",
        **kwargs,
    ):
        """Fit the model (see :meth:`BaseExtendedMMM.fit`).

        Warns on approximate methods: every AR/RW mediator and latent factor
        adds an ``(n_obs,)`` innovation vector, and MAP/ADVI are known-unstable
        for latent-state graphs of that size -- use NUTS for anything
        decision-facing.
        """
        from ...config.enums import FitMethod

        fit_method = (
            method if isinstance(method, FitMethod) else FitMethod(str(method).lower())
        )
        if fit_method is not FitMethod.NUTS:
            has_states = bool(self.config.latent_factors) or any(
                m.dynamics.value != "static" for m in self.config.mediators
            )
            if has_states:
                warnings.warn(
                    "Approximate fits (MAP/ADVI/Pathfinder) are unstable for "
                    "latent-state graphs (per-week innovation vectors); treat "
                    "the result as a smoke check and re-fit with NUTS before "
                    "trusting effects.",
                    stacklevel=2,
                )
        return super().fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            nuts_sampler=nuts_sampler,
            method=method,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Post-fit: exact counterfactual effects
    # ------------------------------------------------------------------

    def _counterfactual_mu(self, multipliers: np.ndarray) -> np.ndarray:
        """Recompute ``mu`` under scaled media with posterior draws held fixed.

        Structure-preserving intervention: AR innovations / factor draws stay
        at their posterior values (same demand shocks, different media).
        Returns (chain, draw, obs).
        """
        self._check_fitted()
        X_cf = self.X_media * np.asarray(multipliers, dtype=float)[None, :]
        with self.model:
            try:
                pm.set_data({"X_media": X_cf})
                ppc = pm.sample_posterior_predictive(
                    self._trace, var_names=["mu"], progressbar=False
                )
            finally:
                pm.set_data({"X_media": self.X_media})
        return np.asarray(ppc.posterior_predictive["mu"].values, dtype=float)

    def _mu_base(self) -> np.ndarray:
        """Posterior draws of mu at the observed media (chain, draw, obs)."""
        return np.asarray(self._trace.posterior["mu"].values, dtype=float)

    def get_mediation_effects(self) -> pd.DataFrame:
        """Per-channel direct / mediated decomposition in original KPI units.

        ``total_effect`` is the EXACT counterfactual contribution (observed
        media vs channel-zeroed, per posterior draw, summed over the window);
        ``direct_effect`` is the in-graph direct term; ``mediated = total -
        direct``. Per-mediator attribution splits the mediated total in
        proportion to the linearized path strengths (labeled approximate).
        """
        self._check_fitted()
        posterior = self._trace.posterior
        mu_base = self._mu_base()

        results = []
        for c, channel in enumerate(self.channel_names):
            mult = np.ones(self.n_channels)
            mult[c] = 0.0
            mu_cf = self._counterfactual_mu(mult)
            total_draws = (mu_base - mu_cf).sum(axis=-1)  # (chain, draw)

            direct_name = f"direct_effect_{channel}"
            if direct_name in posterior:
                direct_draws = np.asarray(
                    posterior[direct_name].values, dtype=float
                ).sum(axis=-1)
            else:
                direct_draws = np.zeros_like(total_draws)

            mediated_draws = total_draws - direct_draws
            total = float(total_draws.mean())
            direct = float(direct_draws.mean())
            mediated = float(mediated_draws.mean())

            # Per-mediator split via linearized path strengths.
            strengths = {}
            for med in self.mediator_names:
                var = f"indirect_{channel}_via_{med}"
                if var in posterior:
                    strengths[med] = float(
                        np.asarray(posterior[var].values, dtype=float).mean()
                    )
            s_total = sum(strengths.values())
            if strengths and abs(s_total) > 1e-12:
                indirect_effects = {
                    med: mediated * (s / s_total) for med, s in strengths.items()
                }
            elif strengths:
                indirect_effects = {med: 0.0 for med in strengths}
            else:
                indirect_effects = {}

            prop = mediated / total if total != 0 else np.nan
            results.append(
                MediationEffects(
                    channel=channel,
                    direct_effect=direct,
                    direct_effect_sd=float(direct_draws.std()),
                    indirect_effects=indirect_effects,
                    total_indirect=mediated,
                    total_effect=total,
                    proportion_mediated=prop,
                )
            )
        return pd.DataFrame([r.to_dict() for r in results])

    def get_channel_roas(self, spend: dict[str, float] | None = None) -> pd.DataFrame:
        """Counterfactual total contribution / spend per channel.

        ``spend`` overrides the per-channel divisor (raw dollars) when the
        modeled media variable is not spend; defaults to the raw ``X_media``
        column sums.
        """
        self._check_fitted()
        mu_base = self._mu_base()
        rows = []
        for c, channel in enumerate(self.channel_names):
            mult = np.ones(self.n_channels)
            mult[c] = 0.0
            mu_cf = self._counterfactual_mu(mult)
            contrib_draws = (mu_base - mu_cf).sum(axis=-1)
            denom = (
                float(spend[channel])
                if spend and channel in spend
                else float(np.asarray(self.X_media)[:, c].sum())
            )
            if denom <= 0:
                roas_mean = roas_sd = np.nan
            else:
                roas_draws = contrib_draws / denom
                roas_mean = float(roas_draws.mean())
                roas_sd = float(roas_draws.std())
            rows.append(
                {
                    "channel": channel,
                    "contribution": float(contrib_draws.mean()),
                    "contribution_sd": float(contrib_draws.std()),
                    "spend": denom,
                    "roas": roas_mean,
                    "roas_sd": roas_sd,
                }
            )
        return pd.DataFrame(rows)

    def get_pathway_effects(self) -> pd.DataFrame:
        """The linearized per-(channel, mediator) path strengths (original KPI
        units per unit saturated media) -- 'how much of TV flows via awareness
        vs consideration'. First-order approximation through sigmoid links and
        AR gains; exact channel totals come from :meth:`get_mediation_effects`."""
        self._check_fitted()
        posterior = self._trace.posterior
        rows = []
        for channel in self.channel_names:
            for med in self.mediator_names:
                var = f"indirect_{channel}_via_{med}"
                if var not in posterior:
                    continue
                draws = np.asarray(posterior[var].values, dtype=float)
                rows.append(
                    {
                        "channel": channel,
                        "mediator": med,
                        "path_strength": float(draws.mean()),
                        "path_strength_sd": float(draws.std()),
                    }
                )
        return pd.DataFrame(rows)


__all__ = ["StructuralNestedMMM"]
