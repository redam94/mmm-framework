"""Bayesian Confirmatory Factor Analysis (Model Garden example) — a **non-MMM**
family that rides the same agent / estimand / serialization pipeline as an MMM.

CFA is a *measurement* model: a set of observed **indicators** is explained by a
small number of **latent factors** with a researcher-specified ("confirmatory")
loading pattern — each indicator loads on exactly one factor. There are no media
channels, no spend, no single KPI. It answers:

    "Do these indicators measure the latent constructs I hypothesized, and how
     strongly does each indicator load on its factor?"

How it plugs in (the non-MMM contract)
--------------------------------------
It subclasses :class:`~mmm_framework.garden.CustomMMM` but declares
``__garden_model_kind__ = "cfa"`` — which exempts it from the MMM-specific garden
gates (channel attributes, ``beta_<channel>`` posterior convention, the channel
read-ops / compat tiers). It overrides only ``_prepare_data`` (assemble the
indicator matrix) and ``_build_model`` (the CFA graph), and reuses ``fit`` /
serialization / the estimand engine unchanged. Its quantities of interest are
expressed as **estimands** over latent posterior variables: per-draw fit indices
(``srmr``, ``cov_fit``) and named scalar loadings — surfaced through the same
``evaluate_estimands`` path as an MMM's ROI.

Identification + likelihood
---------------------------
Factors are standardized (unit variance, uncorrelated), so all loadings are free
and identified up to sign; loadings are given **positive** (HalfNormal) priors —
a *positive-loading* CFA, the common case where each indicator is positively
associated with its construct (planted data here is positive). The likelihood is
the **marginal** multivariate normal (factor scores integrated out):

    yᵢ ~ MvNormal(0, ΛΛᵀ + Ψ),   Ψ = diag(residual_sd²)

which avoids the per-observation latent-score funnel and samples cleanly.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field

from mmm_framework.estimands.registry import factor_loading, fit_index
from mmm_framework.garden import CustomMMM


class CFAConfig(BaseModel):
    """Bespoke, settable configuration for :class:`BayesianCFA` (its CONFIG_SCHEMA).

    ``factor_assignment`` is the confirmatory structure: the 0-based factor index
    each indicator loads on, in indicator (column) order. ``None`` splits the
    indicators evenly across ``n_factors`` (a sensible default for a smoke run).
    """

    n_factors: int = Field(default=2, ge=1)
    factor_assignment: list[int] | None = None
    loading_prior_sigma: float = Field(default=1.0, gt=0)
    standardize_indicators: bool = True

    model_config = {"extra": "forbid"}


class BayesianCFA(CustomMMM):
    """Confirmatory factor analysis as an oracle-compatible non-MMM garden model."""

    __garden_model_kind__ = "cfa"

    #: Bespoke configuration (read via ``self.model_params``).
    CONFIG_SCHEMA = CFAConfig

    #: Fit indices surfaced by default (always present in the posterior). Named
    #: scalar loadings are requested per indicator via ``evaluate_estimands`` and
    #: the full matrix via :meth:`factor_loadings_summary`.
    DEFAULT_ESTIMANDS = [fit_index("srmr"), fit_index("cov_fit")]

    # -- data ----------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Assemble the observed indicator matrix from the panel (all observed
        columns: KPI + media + controls), standardize it, and set the minimal
        model-agnostic attributes the contract/fit/estimand surface reads."""
        frames = [self.panel.y.to_frame()]
        if self.panel.X_media is not None and self.panel.X_media.shape[1] > 0:
            frames.append(self.panel.X_media)
        if self.panel.X_controls is not None and self.panel.X_controls.shape[1] > 0:
            frames.append(self.panel.X_controls)
        import pandas as pd

        observed = pd.concat(frames, axis=1)
        self.indicator_names = [str(c) for c in observed.columns]
        Y = observed.values.astype(np.float64)

        self._ind_mean = Y.mean(axis=0)
        self._ind_std = Y.std(axis=0) + 1e-8
        if self.model_params.standardize_indicators:
            Y = (Y - self._ind_mean) / self._ind_std
        self.indicators = Y
        self.n_obs, self.n_indicators = Y.shape

        # Model-agnostic attributes the base contract / estimand engine read.
        self.channel_names = []  # non-MMM: no channels
        self.control_names = []
        self.n_channels = 0
        self.n_controls = 0
        self._media_raw_max = {}
        self._media_max = {}  # serializer reads this (empty: no channels)
        self.X_controls_raw = None
        self.y = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._scaling_params = {"y_mean": 0.0, "y_std": 1.0}
        self.time_idx = np.arange(self.n_obs)
        # Trend/seasonality are unused by the CFA; empty so the (inherited)
        # serializer + base helpers have something to read.
        self.trend_features = {}
        self.seasonality_features = {}
        self.n_periods = int(getattr(self.panel.coords, "n_periods", self.n_obs))
        self.has_geo = bool(getattr(self.panel.coords, "has_geo", False))
        self.has_product = bool(getattr(self.panel.coords, "has_product", False))

    def _factor_assignment(self) -> np.ndarray:
        cfg = self.model_params
        nf = cfg.n_factors
        if cfg.factor_assignment is not None:
            assign = np.asarray(cfg.factor_assignment, dtype=int)
            if assign.shape[0] != self.n_indicators:
                raise ValueError(
                    f"factor_assignment has {assign.shape[0]} entries but there are "
                    f"{self.n_indicators} indicators {self.indicator_names}"
                )
            if assign.min() < 0 or assign.max() >= nf:
                raise ValueError(f"factor_assignment entries must be in [0, {nf - 1}]")
            return assign
        # Default: split indicators as evenly as possible across the factors.
        return np.array([i % nf for i in range(self.n_indicators)], dtype=int)

    # -- model ---------------------------------------------------------------

    def _build_model(self) -> pm.Model:
        cfg = self.model_params
        Y = self.indicators
        n_ind, nf = self.n_indicators, cfg.n_factors
        assign = self._factor_assignment()
        # One-hot (indicator -> its factor) so the loading matrix is free on the
        # assigned cell and exactly zero elsewhere (confirmatory simple structure).
        onehot = np.zeros((n_ind, nf))
        onehot[np.arange(n_ind), assign] = 1.0

        # Observed covariance (fixed data) — the target the implied covariance fits.
        S = np.cov(Y, rowvar=False)
        diag_s = np.sqrt(np.diag(S))
        srmr_denom = np.outer(diag_s, diag_s)
        offdiag = 1.0 - np.eye(n_ind)

        coords = {
            "obs": np.arange(self.n_obs),
            "indicator": self.indicator_names,
            "factor": [f"F{f + 1}" for f in range(nf)],
        }
        with pm.Model(coords=coords) as model:
            # Positive loadings (sign-identified). One free magnitude per indicator
            # placed in its assigned factor column.
            load_mag = pm.HalfNormal(
                "loading_mag", sigma=cfg.loading_prior_sigma, dims="indicator"
            )
            loadings = pm.Deterministic(
                "factor_loadings",
                load_mag[:, None] * pt.as_tensor_variable(onehot),
                dims=("indicator", "factor"),
            )
            # Named scalar loadings → addressable as `factor_loading` estimands.
            for i, name in enumerate(self.indicator_names):
                pm.Deterministic(f"loading_{name}", load_mag[i])

            residual_sd = pm.HalfNormal("residual_sd", sigma=1.0, dims="indicator")
            implied = pt.dot(loadings, loadings.T) + pt.diag(residual_sd**2)

            # Marginal CFA likelihood: each row ~ MvNormal(0, implied).
            pm.MvNormal(
                "y_obs",
                mu=np.zeros(n_ind),
                cov=implied,
                observed=Y,
                dims=("obs", "indicator"),
            )

            # Per-draw fit indices (model-implied vs observed covariance).
            S_t = pt.as_tensor_variable(S)
            std_resid = (S_t - implied) / pt.as_tensor_variable(srmr_denom)
            pm.Deterministic("srmr", pt.sqrt(pt.mean(std_resid**2)))
            off = pt.as_tensor_variable(offdiag)
            resid_ss = pt.sum(((S_t - implied) ** 2) * off)
            total_ss = float(np.sum((S**2) * offdiag))
            # R²-like covariance fit on the off-diagonal (1 = perfect; honest name,
            # not the χ²-based CFI which needs a null-model fit).
            pm.Deterministic("cov_fit", 1.0 - resid_ss / total_ss)

        return model

    # -- reporting -----------------------------------------------------------

    def factor_loadings_summary(self, hdi_prob: float = 0.94):
        """Posterior loadings table (mean + HDI) per (indicator, factor) — the
        matrix-valued counterpart the scalar estimands don't cover."""
        import arviz as az
        import pandas as pd

        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        da = self._trace.posterior["factor_loadings"]
        mean = da.mean(("chain", "draw")).values
        hdi = az.hdi(self._trace, var_names=["factor_loadings"], hdi_prob=hdi_prob)[
            "factor_loadings"
        ].values  # (indicator, factor, 2)
        rows = []
        assign = self._factor_assignment()
        for i, ind in enumerate(self.indicator_names):
            f = int(assign[i])
            rows.append(
                {
                    "indicator": ind,
                    "factor": f"F{f + 1}",
                    "loading": float(mean[i, f]),
                    "hdi_low": float(hdi[i, f, 0]),
                    "hdi_high": float(hdi[i, f, 1]),
                }
            )
        return pd.DataFrame(rows)


GARDEN_MODEL = BayesianCFA


def synthetic_cfa_panel(n: int = 400, true_load: float = 0.75, seed: int = 7):
    """A :class:`PanelDataset` of 6 indicators with a KNOWN 2-factor structure
    (x1–x3 load on F1, x4–x6 on F2). Authoring convention: indicators are listed
    as the kpi + ``media_channels`` (the CFA treats every observed column as an
    indicator uniformly), in the same order as ``factor_assignment``."""
    import pandas as pd

    from mmm_framework.config import (
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset

    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    cols = {}
    for j in range(3):
        cols[f"x{j + 1}"] = true_load * f1 + rng.normal(scale=0.66, size=n)
    for j in range(3):
        cols[f"x{j + 4}"] = true_load * f2 + rng.normal(scale=0.66, size=n)
    df = pd.DataFrame(cols)
    periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    indicators = list(df.columns)
    media = indicators[1:]
    config = MFFConfig(
        kpi=KPIConfig(name=indicators[0], dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD]) for c in media
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=df[indicators[0]],
        X_media=df[media],
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=media,
            controls=None,
        ),
        index=periods,
        config=config,
    )
    return panel, true_load


if __name__ == "__main__":
    # Smoke test: fit on synthetic data with a KNOWN 2-factor structure and check
    # the recovered loadings + fit indices against the planted truth.
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    panel, true_load = synthetic_cfa_panel()
    print("Fitting BayesianCFA (MAP) on synthetic 2-factor data…")
    mmm = BayesianCFA(
        panel,
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params={"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1]},
    )
    mmm.fit(method="map", random_seed=7)

    print(f"\nTrue loading ≈ {true_load:.2f} on the assigned factor.\n")
    print(mmm.factor_loadings_summary().to_string(index=False))

    est = mmm.evaluate_estimands()  # DEFAULT_ESTIMANDS: srmr, cov_fit
    print("\nFit indices:")
    for name, r in est.items():
        print(f"  {name:10s} mean={r.mean:.3f}  ({r.status})")
