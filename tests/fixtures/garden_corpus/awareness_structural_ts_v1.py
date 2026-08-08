"""Persistent-Awareness Structural Time-Series MMM (Model Garden example).

A bespoke :class:`~mmm_framework.garden.CustomMMM` that re-frames the MMM around
a **latent brand-awareness stock** instead of a contemporaneous sales response.
It answers a different question than the stock model:

    "How does media build *awareness*, and how long does that awareness persist
     after the spend stops?"

Why a structural time series
----------------------------
In the base :class:`~mmm_framework.BayesianMMM`, persistence is a property of
each *channel* (geometric/Weibull adstock on the media input). Here persistence
is a property of *awareness itself*: media pours into a goodwill **stock** that
carries over period-to-period and decays geometrically toward a baseline when
spending pauses. This is the Nerlove–Arrow advertising-goodwill model written
as a Bayesian **local-level state-space** model:

    Latent awareness state (per period ``t``):

        Aₜ  =  intercept                          (equilibrium baseline)
             + Lₜ                                 (organic structural level)
             + Σ_c Sₜ,c                            (media goodwill, by channel)

    Two coupled recursions, both decaying at the SAME persistence ``ρ``:

        Organic level   Lₜ   = ρ·Lₜ₋₁   + εₜ                   εₜ ~ N(0, σ_level)
        Channel stock   Sₜ,c = ρ·Sₜ₋₁,c + βc·sat_c(xₜ,c)       (saturated spend)

    Observation:

        yₜ ~ Normal(Aₜ + seasonality + controls, σ)

``ρ`` (``awareness_retention``, a ``Beta`` on ``(0, 1)``) is the headline
parameter — the share of awareness retained each period. Its half-life
``ln(0.5)/ln(ρ)`` is the answer to "how long does brand memory last?".
Because the channel goodwill ``Sₜ,c`` decays geometrically at ``ρ``, each
channel's carryover IS a geometric adstock with ``alpha = ρ`` — so the standard
adstock / half-life reporting reads true values straight from this model.

Oracle contract
---------------
Subclassing ``CustomMMM`` and overriding only ``_build_model`` keeps the full
agent/reporting/serialization surface working: the override registers the exact
deterministics the read-ops consume (``intercept_component``,
``trend_component`` [= the organic level], ``seasonality_component``,
``channel_contributions`` [= per-channel goodwill stock], ``media_total``,
``controls_total``, ``y_obs_scaled``) plus ``beta_<ch>``, ``sat_*_<ch>``, and an
``adstock_alpha_<ch>`` alias of ``ρ`` for carryover reporting.

Scope
-----
This example models a **single national awareness series** (one stock). It
deliberately raises on geo/product panels rather than silently sharing one stock
across cells — extend it with a per-cell scan to support hierarchical panels.
The KPI should be a persistent brand metric (aided/unaided awareness %, a brand
tracker, consideration), not units sold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from mmm_framework.garden import CustomMMM

# These two live in the same module as BayesianMMM and are the single source of
# truth for the saturation formula + prior-config sampling, so reuse them rather
# than re-deriving (keeps this model's likelihood identical to the base stack's
# and its saturation-curve reporting consistent).
from mmm_framework.model.base import _apply_saturation_pt, _sample_from_prior_config


class AwarenessStructuralMMM(CustomMMM):
    """MMM whose KPI is a *persistent awareness stock* built by media.

    The persistence ``ρ`` is shared by the organic level and every channel's
    goodwill stock, so the model has ONE interpretable "brand memory" knob whose
    half-life is reported directly. Tune the priors below per brand: a slow-decay
    category (insurance, autos) wants a higher-mean ``RETENTION_PRIOR``; an
    impulse category (promotions) a lower one.
    """

    #: ``Beta(α, β)`` prior on the awareness retention ``ρ``. ``Beta(6, 2)`` has
    #: mean 0.75 (≈ 2.4-period half-life on weekly data) — awareness is sticky.
    RETENTION_PRIOR: tuple[float, float] = (6.0, 2.0)

    #: Prior scale of the organic level's weekly innovation. Tight on purpose:
    #: the random-walk baseline should drift slowly so media (not the latent
    #: level) explains the swings and channel effects stay identified.
    LEVEL_INNOVATION_SIGMA: float = 0.15

    def _build_model(self) -> pm.Model:
        """Build the awareness state-space graph (overrides the base MMM build)."""
        if self.has_geo or self.has_product:
            raise NotImplementedError(
                "AwarenessStructuralMMM models a single national awareness "
                "stock and does not support geo/product panels. Aggregate the "
                "data to national, or extend _build_model with a per-cell scan "
                "(carry shape (n_cells, n_channels)). Use the base BayesianMMM "
                "for hierarchical panels."
            )
        if getattr(self, "experiments", None):
            warnings.warn(
                "AwarenessStructuralMMM does not implement in-graph experiment "
                "calibration; registered experiments are ignored by this model.",
                stacklevel=2,
            )

        coords = self._build_coords()
        # Per-channel spend, normalized to ~[0, 1] by each channel's training max
        # — the parametric-path data name `X_media_raw`, so predict() and
        # sample_channel_contributions() can swap it via pm.set_data().
        x_media_norm = self._prepare_raw_media_for_model()
        n_obs = self.n_obs
        # National series ⇒ observation order == period order == time order, so
        # the obs axis IS the time axis the recursions iterate over.
        assert (
            self.time_idx[0] == 0 and self.time_idx[-1] == n_obs - 1
        ), "AwarenessStructuralMMM expects a single time-ordered national series"

        with pm.Model(coords=coords) as model:
            x_media = pm.Data("X_media_raw", x_media_norm, dims=("obs", "channel"))
            if self.X_controls is not None:
                x_controls = pm.Data(
                    "X_controls", self.X_controls, dims=("obs", "control")
                )
            time_idx = pm.Data("time_idx", self.time_idx)

            # --- Equilibrium baseline awareness (the level ρ-decays toward) ----
            intercept = pm.Normal(
                "intercept",
                mu=getattr(self.model_config, "intercept_prior_mu", 0.0),
                sigma=getattr(self.model_config, "intercept_prior_sigma", 0.5),
            )
            pm.Deterministic(
                "intercept_component", intercept + pt.zeros(n_obs), dims="obs"
            )

            # --- Awareness persistence ρ — THE structural knob -----------------
            rho = pm.Beta(
                "awareness_retention",
                alpha=self.RETENTION_PRIOR[0],
                beta=self.RETENTION_PRIOR[1],
            )

            # --- Per-period media increments into the goodwill stock -----------
            # Each channel's saturated, β-weighted spend is the inflow to its
            # goodwill sub-stock this period (diminishing returns via saturation).
            increments = []
            for c, channel in enumerate(self.channel_names):
                sat_kind, sat_params = self._build_channel_saturation(channel)
                x_saturated = _apply_saturation_pt(x_media[:, c], sat_kind, sat_params)

                media_cfg = self.mff_config.get_media_config(channel)
                roi_prior = getattr(media_cfg, "roi_prior", None)
                beta = _sample_from_prior_config(
                    f"beta_{channel}",
                    roi_prior,
                    lambda ch=channel: pm.Gamma(f"beta_{ch}", mu=1.5, sigma=1.0),
                )
                increments.append(beta * x_saturated)  # (n_obs,)

                # The channel's goodwill decays geometrically at ρ, so its
                # carryover IS a geometric adstock with alpha = ρ. Alias it under
                # the convention the adstock/half-life reporting reads.
                pm.Deterministic(f"adstock_alpha_{channel}", rho)

            # --- Organic structural level innovations εₜ ~ N(0, σ_level) -------
            sigma_level = pm.HalfNormal(
                "awareness_level_sigma", sigma=self.LEVEL_INNOVATION_SIGMA
            )
            level_innovation = pm.Normal(
                "awareness_innovation", mu=0.0, sigma=sigma_level, dims="obs"
            )

            # --- Geometric-decay accumulation, VECTORIZED (no pytensor.scan) ----
            # The recursion stateₜ = ρ·stateₜ₋₁ + inflowₜ (state₋₁ = 0) has the
            # closed form  stateₜ = Σ_{τ≤t} ρ^(t-τ)·inflow_τ — a lower-triangular
            # Toeplitz convolution. Building that decay matrix once and matmul-ing
            # is mathematically identical to scanning, but compiles instantly and
            # differentiates cheaply: scan's gradient graph is the slow,
            # GIL-holding step that makes an in-process MAP/NUTS fit crawl (and
            # can freeze a co-hosted API). O(T²) memory — negligible for realistic
            # awareness horizons.
            t = np.arange(n_obs)
            lag = t[:, None] - t[None, :]  # lag[t, τ] = t - τ
            causal = pt.as_tensor_variable(lag >= 0)  # keep τ ≤ t (lower triangle)
            lag_clamped = pt.as_tensor_variable(np.maximum(lag, 0))
            decay = pt.where(causal, rho**lag_clamped, 0.0)  # (n_obs, n_obs)

            media_inflow = pt.stack(increments, axis=1)  # (n_obs, n_ch)
            channel_stock = decay @ media_inflow  # (n_obs, n_ch): goodwill by channel
            organic_level = decay @ level_innovation  # (n_obs,): structural level

            # The organic awareness level is this model's "trend" component.
            pm.Deterministic("trend_component", organic_level, dims="obs")
            pm.Deterministic(
                "channel_contributions", channel_stock, dims=("obs", "channel")
            )
            media_total = channel_stock.sum(axis=1)
            pm.Deterministic("media_total", media_total)

            # --- Seasonality (reused; zero unless the spec configures Fourier) -
            seasonality_at_periods = pt.zeros(self.n_periods)
            for name, features in self.seasonality_features.items():
                season_coef = pm.Normal(
                    f"season_{name}",
                    mu=0,
                    sigma=self.seasonality_config.prior_sigma_for(name),
                    shape=features.shape[1],
                    dims=f"{name}_fourier",
                )
                seasonality_at_periods = seasonality_at_periods + pt.dot(
                    pt.as_tensor_variable(features), season_coef
                )
            seasonality = seasonality_at_periods[time_idx]
            pm.Deterministic("seasonality_component", seasonality)
            pm.Deterministic("seasonality_by_period", seasonality_at_periods)

            # --- Controls (reused base machinery, incl. confounder handling) ---
            sigma = (
                pm.HalfNormal("sigma", sigma=0.5) if self._selection_active() else None
            )
            if self.n_controls > 0:
                beta_controls = self._build_control_betas(sigma)
                control_contribution = pt.dot(x_controls, beta_controls)
                pm.Deterministic(
                    "control_contributions",
                    x_controls * beta_controls,
                    dims=("obs", "control"),
                )
            else:
                control_contribution = pt.zeros(n_obs)
            pm.Deterministic("controls_total", control_contribution, dims="obs")

            # --- Observed awareness = full state + seasonality + controls ------
            mu = (
                intercept
                + organic_level
                + seasonality
                + media_total
                + control_contribution
            )
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=0.5)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y, dims="obs")
            pm.Deterministic(
                "y_obs_scaled", y_obs * self.y_std + self.y_mean, dims="obs"
            )

        return model


# Disambiguate for the Model Garden loader (a module may define helper classes).
GARDEN_MODEL = AwarenessStructuralMMM


if __name__ == "__main__":
    # Standalone smoke test: fit on a synthetic national world and report the
    # learned awareness persistence + half-life + per-channel goodwill ROI.
    import tempfile
    from pathlib import Path

    from mmm_framework import load_mff
    from mmm_framework.agents.fitting import build_model
    from mmm_framework.synth import generate_mff

    print("Generating a synthetic national world ('realistic')…")
    df, answer = generate_mff("realistic", seed=7, n_weeks=104)
    channels = list(answer["channels"])
    controls = [
        v
        for v in dict.fromkeys(df["VariableName"].tolist())
        if v != "Sales" and v not in channels
    ]
    spec = {
        "kpi": "Sales",  # treat the synthetic KPI as the awareness metric
        "media_channels": [{"name": c} for c in channels],
        "control_variables": [{"name": c} for c in controls],
        "trend": {"type": "none"},  # the structural level IS the trend here
        "seasonality": {"yearly": 0, "monthly": 0, "weekly": 0},
        "inference": {"method": "map", "chains": 1, "draws": 200, "tune": 200},
    }

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "synth.csv"
        df.to_csv(csv_path, index=False)
        mmm = build_model(spec, str(csv_path), model_cls=AwarenessStructuralMMM)
        print(f"Fitting AwarenessStructuralMMM on {mmm.n_obs} weeks (MAP)…")
        mmm.fit(method="map", random_seed=7)

        rho = float(mmm._trace.posterior["awareness_retention"].values.mean())
        half_life = np.log(0.5) / np.log(rho) if 0 < rho < 1 else float("inf")
        print(
            f"\nAwareness retention ρ ≈ {rho:.3f}  →  half-life ≈ {half_life:.1f} weeks"
        )

        from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

        roi = compute_roi_with_uncertainty(mmm, hdi_prob=0.94)
        print("\nAwareness-building efficiency (goodwill per $):")
        print(
            roi[["channel", "roi_mean", "roi_hdi_low", "roi_hdi_high"]].to_string(
                index=False
            )
        )
