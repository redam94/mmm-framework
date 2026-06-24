"""Latent-Factor MMM (Model Garden example) — a JOINT model that estimates a
latent construct from many indicators *and* an MMM in one PyMC graph, with the
latent factor entering the MMM mean as a covariate.

The worked example is **economic health**: a handful of macro indicators
(GDP growth, consumer confidence, unemployment, retail sales) are noisy
manifestations of one latent "economic health" factor. Economic health is a
*common cause* — a boom raises both ad budgets and sales — so an MMM that
ignores it leaves the back-door ``spend ← economic health → sales`` open and
over-credits media ROI.

Why a JOINT model (not a 2-stage chain)
---------------------------------------
The naive fix would be a two-stage chain: fit a factor model, take the latent's
posterior *mean*, and plug it into the MMM as a fixed covariate. That is the
classic "generated regressor" — it throws away the factor's uncertainty and
understates the MMM's. Here the measurement block and the MMM block share ONE
latent variable in ONE graph, so:

    indicators ──▶  economic_health  ◀── KPI (via beta_econ)
                         │
                         ▼
                  conditions the media betas (closes the back-door)

The factor is informed by both the indicators and the KPI residual structure,
and its uncertainty propagates into the media coefficients automatically. This
is a sibling of :class:`AwarenessStructuralMMM` (which estimates a latent
goodwill *stock* jointly inside the MMM); we swap the stock for a latent-factor
*measurement* block.

Structure
---------
* MEASUREMENT block (per period ``t``):

    economic_healthₜ  (a per-period latent factor)
    indicatorₖ,ₜ     ~ Normal(λₖ·economic_healthₜ + aₖ, ψₖ)

  ``factor_dynamics="static"`` (default): an iid ``Normal(0, 1)`` factor per
  period (CFA-style). ``"ar1"``: a persistent factor via the scan-free AR(1)
  construction ``economic_healthₜ = ρ·economic_healthₜ₋₁ + εₜ``.

  Identification (THREE fixes, all load-bearing): (1) the realized factor is
  **standardized to unit variance in-graph**, so its scale is fixed and the
  loadings carry the identified indicator↔factor correlations (without it the
  AR(1) variance ``1/(1−ρ²)`` trades off against the loadings and they can
  collapse toward zero); (2) the first (anchor) loading is **positive** (fixes
  orientation/sign), the rest free — so a genuinely negative indicator (e.g.
  unemployment) recovers a negative loading; (3) NUTS is **warm-started at the
  indicator PCA** (:meth:`LatentFactorMMM.suggest_initvals`, injected
  automatically for the ``static`` factor) — a free per-period factor is
  multimodal, and seeding the measurement optimum keeps every chain in the
  indicator-dominated basin.

* MMM block (the standard graph) PLUS ``beta_econ · economic_health[t]``:

    yₜ ~ Normal(intercept + seasonality + Σ media + Σ controls
                + beta_econ·economic_healthₜ , σ)

Scope
-----
The latent factor is a single **national** series (one macro construct),
broadcast across geos via ``time_idx``; the indicators must be national (one
value per period — enforced by collapsing to the period axis in
``_prepare_data``). It is an MMM-kind family (``__garden_model_kind__ == "mmm"``)
so the full channel/ROI/decomposition reporting stays on; it ALSO exposes
``factor_loadings_summary`` so the report renders a factor-loadings section
alongside the MMM sections.

Data contract
-------------
The economic indicators are tagged ``DatasetRole.INDICATOR`` (not ``CONTROL``)
in the role-tagged dataset, so they feed the measurement block and are NOT
silently used as MMM regression controls. ``REQUIRED_ROLES`` enforces this —
a plain MFF panel (which cannot carry indicators) fails fast at construction.
``model_params.indicator_columns`` is a by-name fallback.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field

from mmm_framework.config.roles import DatasetRole
from mmm_framework.estimands.registry import factor_loading, latent_scalar
from mmm_framework.garden import CustomMMM

# Single source of truth for the saturation formula + prior-config sampling, so
# this model's media likelihood is identical to the base stack's (and its
# saturation/ROI reporting consistent).
from mmm_framework.model.base import _apply_saturation_pt, _sample_from_prior_config


class LatentFactorParams(BaseModel):
    """Bespoke, settable, defaulted configuration for :class:`LatentFactorMMM`
    (its ``CONFIG_SCHEMA``). Read off ``self.model_params`` in ``_build_model``."""

    #: ``"static"`` (default) — an iid ``Normal(0, 1)`` factor per period (CFA-
    #: style). It is the robust choice: a free per-period factor is multimodal, and
    #: the static factor can be **warm-started at the indicator PCA** (see
    #: :meth:`LatentFactorMMM.suggest_initvals`) so every NUTS chain lands in the
    #: indicator-dominated basin. With smooth indicators it recovers a smooth
    #: factor anyway. ``"ar1"`` — a persistent factor via the scan-free AR(1)
    #: construction of :class:`AwarenessStructuralMMM`; more principled for a sticky
    #: macro series, but its innovations aren't directly warm-startable, so it needs
    #: heavier tuning and can be unreliable seed-to-seed. Prefer ``"static"``.
    factor_dynamics: Literal["static", "ar1"] = "static"

    #: ``Beta(α, β)`` prior on the AR(1) persistence ρ. ``Beta(5, 2)`` has mean
    #: ≈ 0.71 — a slow macro cycle. Ignored when ``factor_dynamics == "static"``.
    ar_rho_prior_alpha: float = Field(default=5.0, gt=0)
    ar_rho_prior_beta: float = Field(default=2.0, gt=0)

    #: Prior scale of the factor loadings. The anchor loading is ``HalfNormal``
    #: (positive, fixes the factor's orientation); the rest are ``Normal`` (free
    #: sign, so a negatively-related indicator recovers a negative loading).
    loading_prior_sigma: float = Field(default=1.0, gt=0)

    #: Prior scale of each indicator's idiosyncratic residual SD (``HalfNormal``).
    indicator_residual_sigma: float = Field(default=1.0, gt=0)

    #: Prior scale of ``beta_econ`` — the latent factor's effect on the KPI, on
    #: the standardized-y scale.
    beta_econ_prior_sigma: float = Field(default=1.0, gt=0)

    #: By-name fallback for the indicator columns when NOT using the ``INDICATOR``
    #: role (e.g. an ad-hoc panel). Empty → read by role (recommended).
    indicator_columns: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class LatentFactorMMM(CustomMMM):
    """An MMM with a latent factor (e.g. economic health) estimated jointly from
    indicator columns and used as a KPI covariate.

    Its bespoke parameters live in :class:`LatentFactorParams` (the
    ``CONFIG_SCHEMA``). It overrides only ``_prepare_data`` (extract + standardize
    the indicator matrix) and ``_build_model`` (the joint graph); ``fit`` /
    serialization / the estimand engine / reporting are inherited.
    """

    #: It IS an MMM (channels, spend, ROI) — keep the kind ``"mmm"`` so the full
    #: MMM reporting / compat tiers apply. The factor-loadings report section is
    #: turned on by the (duck-typed) presence of ``factor_loadings_summary``, not
    #: by the model kind. See ``reporting/extractors/bayesian.py``.
    __garden_model_kind__ = "mmm"

    #: Bespoke, defaulted, validated configuration (read via ``self.model_params``).
    CONFIG_SCHEMA = LatentFactorParams

    #: Data contract: an outcome + ≥1 media predictor + ≥1 economic INDICATOR.
    #: ``REQUIRED_DATASET_CAPABILITIES`` makes the indicator requirement explicit
    #: (a plain MFF panel — no indicator role — fails fast at construction).
    REQUIRED_ROLES = (
        DatasetRole.TARGET,
        DatasetRole.PREDICTOR,
        DatasetRole.INDICATOR,
    )
    REQUIRED_DATASET_CAPABILITIES = ("HAS_INDICATORS",)

    #: Title for the factor-loadings report section (read host-side by the
    #: Bayesian extractor when populating the latent-structure bundle fields).
    LATENT_SECTION_TITLE = "Economic Health Factor"

    # -- data ----------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Build the normal MMM views (y / X_media / X_controls) via the base, then
        additionally extract the economic-indicator matrix by ROLE.

        The indicators are tagged ``INDICATOR`` (not ``CONTROL``), so the base's
        ``X_controls`` view excludes them — they cannot leak into the MMM
        regression. They are standardized (loadings live on the standardized
        scale) and collapsed to one row per period (the factor is national).
        """
        super()._prepare_data()  # y, X_media, X_controls, time_idx, geo, …

        ind = self.dataset.frame_for(DatasetRole.INDICATOR)
        if self.model_params.indicator_columns:  # by-name fallback
            ind = self.dataset.table[self.model_params.indicator_columns]
        self.indicator_names = [str(c) for c in ind.columns]
        Z = ind.to_numpy(dtype=np.float64)
        if Z.shape[1] == 0:
            raise ValueError(
                "LatentFactorMMM needs >=1 column tagged DatasetRole.INDICATOR "
                "(or model_params.indicator_columns). The economic indicators must "
                "be tagged INDICATOR, not CONTROL, or they would be silently used "
                "as MMM regression controls."
            )

        self._ind_mean = Z.mean(axis=0)
        self._ind_std = Z.std(axis=0) + 1e-8
        self.indicators = (Z - self._ind_mean) / self._ind_std  # (n_obs, n_ind)
        self.n_indicators = self.indicators.shape[1]

        # Collapse to one row per PERIOD for the measurement block: the factor is
        # a single national series, so a geo panel must not feed G replicate
        # indicator rows per period (that would G-fold over-weight the measurement
        # block). For a national series (n_obs == n_periods) this is the identity.
        byp = np.zeros((self.n_periods, self.n_indicators))
        counts = np.zeros(self.n_periods)
        np.add.at(byp, self.time_idx, self.indicators)
        np.add.at(counts, self.time_idx, 1.0)
        self.indicators_by_period = byp / np.maximum(counts, 1.0)[:, None]

    # -- model ---------------------------------------------------------------

    def _build_model(self) -> pm.Model:
        """Build the joint measurement + MMM graph (overrides the base build)."""
        p = self.model_params
        n_obs = self.n_obs
        n_ind = self.n_indicators

        coords = self._build_coords()
        coords["indicator"] = self.indicator_names
        coords["period"] = list(range(self.n_periods))

        # Parametric (in-graph) adstock path: feed the normalized raw spend; the
        # per-channel kernel RVs (incl. ``adstock_alpha_<ch>``) are created in
        # ``_channel_adstock_apply`` so carryover/half-life reporting reads true.
        x_media_norm = self._prepare_raw_media_for_model()

        with pm.Model(coords=coords) as model:
            x_media = pm.Data("X_media_raw", x_media_norm, dims=("obs", "channel"))
            if self.X_controls is not None:
                x_controls = pm.Data(
                    "X_controls", self.X_controls, dims=("obs", "control")
                )
            time_idx = pm.Data("time_idx", self.time_idx)

            # =================================================================
            # BLOCK A — MEASUREMENT: the latent economic-health factor
            # =================================================================
            if p.factor_dynamics == "ar1":
                # AR(1) on the PERIOD axis, non-centered, scan-free (the
                # AwarenessStructuralMMM decay-matrix trick): the recursion
                # fₜ = ρ·fₜ₋₁ + εₜ (f₋₁ = 0) has the closed form
                # fₜ = Σ_{τ≤t} ρ^(t-τ)·ε_τ — a lower-triangular Toeplitz matmul.
                # Innovation variance fixed to 1 -> the factor's scale is
                # determined (loadings + beta_econ carry the units).
                rho = pm.Beta(
                    "econ_persistence",
                    alpha=p.ar_rho_prior_alpha,
                    beta=p.ar_rho_prior_beta,
                )
                eps = pm.Normal("econ_innovation", 0.0, 1.0, dims="period")
                tau = np.arange(self.n_periods)
                lag = tau[:, None] - tau[None, :]
                causal = pt.as_tensor_variable(lag >= 0)
                lag_clamped = pt.as_tensor_variable(np.maximum(lag, 0))
                decay = pt.where(causal, rho**lag_clamped, 0.0)
                econ_period = decay @ eps  # (n_periods,)
            else:  # static: iid standard-normal factor per period (CFA-style)
                econ_period = pm.Normal("econ_innovation", 0.0, 1.0, dims="period")

            # Standardize the realized factor to unit empirical variance. This PINS
            # the factor's scale, so the loadings carry the identified
            # indicator↔factor correlations and are decoupled from ρ. Without it the
            # AR(1) variance 1/(1−ρ²) trades off against the loadings and the
            # sampler can collapse them toward zero (the factor drifts as noise);
            # ρ still sets the factor's shape/persistence.
            econ_std = (econ_period - econ_period.mean()) / (econ_period.std() + 1e-6)
            economic_health = pm.Deterministic(
                "economic_health", econ_std, dims="period"
            )
            econ_obs = economic_health[time_idx]  # (n_obs,) — geo-safe broadcast
            pm.Deterministic("economic_health_obs", econ_obs, dims="obs")

            # Loadings: the FIRST (anchor) indicator loads positively (orientation
            # / sign identification); the rest are free-sign, so a negatively
            # related indicator (e.g. unemployment) recovers a negative loading.
            sigma_load = p.loading_prior_sigma
            lam0 = pm.HalfNormal("loading_anchor", sigma=sigma_load)
            if n_ind > 1:
                lam_rest = pm.Normal(
                    "loading_rest", mu=0.0, sigma=sigma_load, shape=n_ind - 1
                )
                load_vec = pt.concatenate([lam0[None], lam_rest])
            else:
                load_vec = lam0[None]
            loadings = pm.Deterministic("factor_loadings", load_vec, dims="indicator")
            # Named scalar loadings -> addressable as `factor_loading` estimands.
            for i, name in enumerate(self.indicator_names):
                pm.Deterministic(f"loading_{name}", loadings[i])

            ind_intercept = pm.Normal(
                "indicator_intercept", mu=0.0, sigma=0.5, dims="indicator"
            )
            ind_sigma = pm.HalfNormal(
                "indicator_sigma", sigma=p.indicator_residual_sigma, dims="indicator"
            )
            # indicatorₖ,ₜ ~ Normal(λₖ·economic_healthₜ + aₖ, ψₖ), on the PERIOD
            # axis (standardized indicators collapsed to one row per period).
            ind_mu = (
                economic_health[:, None] * loadings[None, :] + ind_intercept[None, :]
            )
            pm.Normal(
                "indicator_obs",
                mu=ind_mu,
                sigma=ind_sigma,
                observed=self.indicators_by_period,
                dims=("period", "indicator"),
            )

            # =================================================================
            # BLOCK B — MMM: standard adstock -> saturation -> regression
            # =================================================================
            intercept = pm.Normal(
                "intercept",
                mu=getattr(self.model_config, "intercept_prior_mu", 0.0),
                sigma=getattr(self.model_config, "intercept_prior_sigma", 0.5),
            )
            pm.Deterministic(
                "intercept_component", intercept + pt.zeros(n_obs), dims="obs"
            )
            # This model has no separate parametric trend — the latent factor is
            # the slow signal. Register a zero trend for the read-op contract.
            pm.Deterministic("trend_component", pt.zeros(n_obs), dims="obs")

            channel_contribs = []
            for c, channel in enumerate(self.channel_names):
                adstock_apply, _ = self._channel_adstock_apply(channel)
                x_adstocked = adstock_apply(x_media[:, c])
                sat_kind, sat_params = self._build_channel_saturation(channel)
                x_saturated = _apply_saturation_pt(x_adstocked, sat_kind, sat_params)
                media_cfg = self.mff_config.get_media_config(channel)
                roi_prior = getattr(media_cfg, "roi_prior", None)
                beta = _sample_from_prior_config(
                    f"beta_{channel}",
                    roi_prior,
                    lambda ch=channel: pm.Gamma(f"beta_{ch}", mu=1.5, sigma=1.0),
                )
                channel_contribs.append(beta * x_saturated)

            media_matrix = pt.stack(channel_contribs, axis=1)
            pm.Deterministic(
                "channel_contributions", media_matrix, dims=("obs", "channel")
            )
            media_total = media_matrix.sum(axis=1)
            pm.Deterministic("media_total", media_total)

            # Seasonality (reused; zero unless the spec configures Fourier).
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

            # Controls (reused base machinery, incl. confounder handling).
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

            # --- THE COUPLING: the latent factor enters the KPI mean ----------
            beta_econ = pm.Normal(
                "beta_economic_health", mu=0.0, sigma=p.beta_econ_prior_sigma
            )
            econ_contribution = beta_econ * econ_obs
            pm.Deterministic(
                "economic_health_contribution", econ_contribution, dims="obs"
            )

            mu = (
                intercept
                + seasonality
                + media_total
                + control_contribution
                + econ_contribution
            )
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=0.5)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y, dims="obs")
            pm.Deterministic(
                "y_obs_scaled", y_obs * self.y_std + self.y_mean, dims="obs"
            )

        return model

    # -- fitting (warm start) -----------------------------------------------

    def suggest_initvals(self) -> dict:
        """NUTS warm-start at the indicator PCA — the cure for factor-model
        multimodality.

        A free per-period latent factor has rotation / local modes: from an
        unlucky init a chain latches onto one indicator's noise (loadings collapse
        toward zero, the factor decorrelates from the truth). The first principal
        component of the indicator block IS the measurement model's optimum and
        carries far more observations (``n_ind × n_periods``) than the single KPI
        series, so seeding every chain there keeps them in the right basin. Only
        the ``static`` factor is directly initialisable (the AR(1) ``eps`` would
        need the unknown ρ to invert); returns just the loading inits otherwise.
        """
        Z = self.indicators_by_period
        u, _s, _vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        pc1 = u[:, 0]
        pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-9)
        loads = np.array(
            [np.corrcoef(Z[:, k], pc1)[0, 1] for k in range(self.n_indicators)]
        )
        if loads[0] < 0:  # align the factor's sign to the positive anchor (col 0)
            pc1, loads = -pc1, -loads
        init: dict = {"loading_anchor": float(abs(loads[0]))}
        if self.n_indicators > 1:
            init["loading_rest"] = loads[1:].astype(np.float64)
        if self.model_params.factor_dynamics == "static":
            init["econ_innovation"] = pc1.astype(np.float64)
        return init

    def fit(self, *args, **kwargs):
        """Fit, auto-warm-starting NUTS at the indicator PCA (see
        :meth:`suggest_initvals`).

        The warm start is injected only when the caller runs NUTS (the default),
        did not pass their own ``initvals``, and the factor is ``static`` (the
        directly-initialisable path). Approximate fits (``method="map"``/VI) and
        AR(1) are left untouched."""
        method = kwargs.get("method")
        is_nuts = method is None or str(getattr(method, "value", method)).lower() in (
            "nuts",
            "",
        )
        if (
            is_nuts
            and "initvals" not in kwargs
            and self.model_params.factor_dynamics == "static"
        ):
            try:
                kwargs["initvals"] = self.suggest_initvals()
            except Exception:  # noqa: BLE001 — warm start is best-effort
                pass
        return super().fit(*args, **kwargs)

    # -- estimands + reporting ----------------------------------------------

    def _default_estimands(self) -> list:
        """De-biased media ROI (the point) + the latent factor's level + one named
        loading estimand per indicator. Built dynamically because the indicator
        names are known only after ``_prepare_data`` (mirrors LCA's class-size
        estimands)."""
        ests = [
            "contribution_roi",
            "marginal_roas",
            latent_scalar(
                "economic_health_level",
                var="economic_health_obs",
                kind="latent_state",
                units="factor (std)",
                causal_assumptions=(
                    "Mean per-period level of the latent economic-health factor."
                ),
            ),
        ]
        ests += [
            factor_loading(f"loading_{name}", var=f"loading_{name}")
            for name in getattr(self, "indicator_names", [])
        ]
        return [self._resolve_estimand(e) for e in ests]

    def factor_loadings_summary(self, hdi_prob: float = 0.94):
        """Posterior loadings table (mean + HDI) per indicator — the matrix-valued
        counterpart the scalar estimands don't cover. Drives the report's factor
        section (duck-typed by the Bayesian extractor)."""
        import arviz as az
        import pandas as pd

        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        da = self._trace.posterior["factor_loadings"]
        mean = da.mean(("chain", "draw")).values  # (indicator,)
        hdi = az.hdi(self._trace, var_names=["factor_loadings"], hdi_prob=hdi_prob)[
            "factor_loadings"
        ].values  # (indicator, 2)
        rows = []
        for i, ind in enumerate(self.indicator_names):
            rows.append(
                {
                    "indicator": ind,
                    "factor": "EconomicHealth",
                    "loading": float(mean[i]),
                    "hdi_low": float(hdi[i, 0]),
                    "hdi_high": float(hdi[i, 1]),
                }
            )
        return pd.DataFrame(rows)


# Disambiguate for the Model Garden loader (a module may define helper classes).
GARDEN_MODEL = LatentFactorMMM


# ---------------------------------------------------------------------------
# Worked-example data: a role-tagged Dataset where economic indicators are
# tagged INDICATOR (so they feed the measurement block, not the MMM controls).
# ---------------------------------------------------------------------------


def economic_health_dataset(seed: int = 14, n_weeks: int | None = None):
    """Build a role-tagged :class:`Dataset` for :class:`LatentFactorMMM` from the
    ``economic_health`` synthetic world, returning ``(dataset, scenario)``.

    KPI → TARGET, channels → PREDICTOR, Price → CONTROL, and the 4 economic
    indicators → INDICATOR (gdp_growth first, as the positive anchor).
    """
    import pandas as pd

    from mmm_framework.config.dataset import DatasetSchema, RoleBinding
    from mmm_framework.dataset import Dataset
    from mmm_framework.synth import dgp

    sc = dgp.make_economic_health(seed=seed, n_weeks=n_weeks)
    indicators = sc.notes["indicators"]  # {name: array}, gdp_growth first

    table = pd.DataFrame({"Period": sc.weeks})
    table["Sales"] = sc.y.to_numpy()
    for c in sc.channels:
        table[c] = sc.spend[c].to_numpy()
    table["Price"] = sc.controls["Price"].to_numpy()
    for name, arr in indicators.items():
        table[name] = np.asarray(arr)

    bindings = [RoleBinding(name="Sales", role=DatasetRole.TARGET)]
    bindings += [RoleBinding(name=c, role=DatasetRole.PREDICTOR) for c in sc.channels]
    bindings.append(RoleBinding(name="Price", role=DatasetRole.CONTROL))
    bindings += [
        RoleBinding(name=name, role=DatasetRole.INDICATOR) for name in indicators
    ]
    schema = DatasetSchema(bindings=bindings, time_col="Period", frequency="W")
    return Dataset.from_wide(table, schema), sc


if __name__ == "__main__":
    # Standalone smoke test: fit on the economic-health world (NUTS, auto
    # warm-started) and report the recovered loadings, the de-biased media ROI,
    # and the declared estimands.
    import numpy as np

    from mmm_framework.config import ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    print("Building the economic-health world (latent factor + 4 indicators)…")
    dataset, sc = economic_health_dataset(seed=14)
    mmm = LatentFactorMMM(dataset, ModelConfig(), TrendConfig(type=TrendType.NONE))
    print(f"Fitting LatentFactorMMM on {mmm.n_obs} weeks (NUTS, PCA warm-start)…")
    mmm.fit(draws=400, tune=800, chains=4, target_accept=0.9, random_seed=7)

    print("\nRecovered factor loadings (truth in parens):")
    truth = sc.notes["true_loadings"]
    summary = mmm.factor_loadings_summary()
    for _, r in summary.iterrows():
        print(
            f"  {r['indicator']:<20} loading={r['loading']:+.2f}  "
            f"(true {truth[r['indicator']]:+.2f})"
        )

    E_hat = mmm._trace.posterior["economic_health"].mean(("chain", "draw")).values
    r = np.corrcoef(E_hat, sc.notes["latent_econ"])[0, 1]
    print(f"\ncorr(recovered factor, truth) = {abs(r):.3f}")

    from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

    roi = compute_roi_with_uncertainty(mmm, hdi_prob=0.94)
    print("\nDe-biased media ROI (true_roas in parens):")
    for _, row in roi.iterrows():
        ch = row["channel"]
        print(f"  {ch:<10} roi={row['roi_mean']:+.3f}  (true {sc.true_roas[ch]:.3f})")

    print("\nDeclared estimands (model-realized):")
    for key, res in mmm.evaluate_estimands().items():
        mean = "—" if res.mean is None else f"{res.mean:+.3f}"
        print(f"  {key:<28} mean={mean:>8}  status={res.status}")
