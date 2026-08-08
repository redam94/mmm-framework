"""Bayesian Latent Class Regression (Model Garden) — extends LCA by allowing
consumer demographics (age, income, etc.) to predict segment membership.

If demographic covariates are provided in `X_controls`, the model automatically
switches from a static mixture to a Latent Class Regression. It uses a multinomial
logit link where Class 1 acts as the reference category (coefficients pinned to 0).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field
from pymc.distributions.transforms import ordered

from mmm_framework.estimands.registry import latent_scalar
from mmm_framework.garden import CustomMMM


class LCAConfig(BaseModel):
    """Bespoke, settable configuration for :class:`BayesianLCA`."""

    n_classes: int = Field(default=2, ge=2)
    #: Beta(a, b) prior on each item-endorsement probability θₖⱼ.
    item_prior_a: float = Field(default=1.0, gt=0)
    item_prior_b: float = Field(default=1.0, gt=0)
    #: Spread of the ordered class-logit prior.
    class_logit_sigma: float = Field(default=1.5, gt=0)
    #: Prior standard deviation for the demographic covariate effects.
    covariate_sigma: float = Field(default=1.0, gt=0)
    #: Indicators that aren't already 0/1 are thresholded at this value.
    binarize_threshold: float | None = None

    model_config = {"extra": "forbid"}


class BayesianLCR(CustomMMM):
    """Latent class regression model supporting demographic covariates."""

    __garden_model_kind__ = "latent_class"
    CONFIG_SCHEMA = LCAConfig

    # -- data ----------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Assemble the binary purchase indicators from KPI + Media, and extract
        demographic covariates from Controls."""
        import pandas as pd

        # 1. Assemble the binary purchase items (Y)
        frames = [self.panel.y.to_frame()]
        if self.panel.X_media is not None and self.panel.X_media.shape[1] > 0:
            frames.append(self.panel.X_media)
        observed = pd.concat(frames, axis=1)
        self.item_names = [str(c) for c in observed.columns]
        Y = observed.values.astype(np.float64)

        thr = self.model_params.binarize_threshold
        if thr is not None:
            Y = (Y > float(thr)).astype(np.float64)
        elif not np.isin(np.unique(Y), (0.0, 1.0)).all():
            Y = (Y > np.median(Y, axis=0)).astype(np.float64)
        self.items = Y
        self.n_obs, self.n_items = Y.shape

        # 2. Extract and standardize demographic covariates (X)
        if self.panel.X_controls is not None and self.panel.X_controls.shape[1] > 0:
            self.covariate_names = [str(c) for c in self.panel.X_controls.columns]
            X_raw = self.panel.X_controls.values.astype(np.float64)
            # Standardize continuous covariates for stable MCMC sampling
            self.X_mean = np.mean(X_raw, axis=0)
            self.X_std = np.std(X_raw, axis=0)
            self.X_std[self.X_std == 0.0] = 1.0  # avoid division by zero
            self.covariates = (X_raw - self.X_mean) / self.X_std
            self.n_covariates = self.covariates.shape[1]
            self.has_covariates = True
        else:
            self.covariate_names = []
            self.covariates = np.empty((self.n_obs, 0))
            self.n_covariates = 0
            self.has_covariates = False

        # Model-agnostic attributes for the base contract
        self.channel_names = []
        self.control_names = self.covariate_names
        self.n_channels = 0
        self.n_controls = self.n_covariates
        self._media_raw_max = {}
        self._media_max = {}
        self.X_controls_raw = self.panel.X_controls
        self.y = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._scaling_params = {"y_mean": 0.0, "y_std": 1.0}
        self.time_idx = np.arange(self.n_obs)
        self.trend_features = {}
        self.seasonality_features = {}
        self.n_periods = int(getattr(self.panel.coords, "n_periods", self.n_obs))
        self.has_geo = bool(getattr(self.panel.coords, "has_geo", False))
        self.has_product = bool(getattr(self.panel.coords, "has_product", False))

    # -- model ---------------------------------------------------------------

    def _build_model(self) -> pm.Model:
        cfg = self.model_params
        Y = self.items
        K, J = cfg.n_classes, self.n_items

        coords = {
            "obs": np.arange(self.n_obs),
            "klass": [f"C{k + 1}" for k in range(K)],
            "item": self.item_names,
        }
        if self.has_covariates:
            coords["covariate"] = self.covariate_names
            coords["klass_minus_one"] = [f"C{k + 1}" for k in range(1, K)]

        with pm.Model(coords=coords) as model:
            # 1. Class Intercepts (ordered to prevent label switching)
            intercepts = pm.Normal(
                "class_intercepts",
                mu=0.0,
                sigma=cfg.class_logit_sigma,
                shape=K,
                transform=ordered,
                initval=np.linspace(-1.0, 1.0, K),
                dims="klass",
            )

            # 2. Demographic Covariate Effects (Beta)
            if self.has_covariates:
                X_tensor = pt.as_tensor_variable(self.covariates)
                # Reference class (C1) has effects fixed at 0 for identification
                beta_raw = pm.Normal(
                    "beta_cov_raw",
                    mu=0.0,
                    sigma=cfg.covariate_sigma,
                    shape=(K - 1, self.n_covariates),
                    dims=("klass_minus_one", "covariate"),
                )
                beta = pm.Deterministic(
                    "beta_cov",
                    pt.concatenate(
                        [pt.zeros((1, self.n_covariates)), beta_raw], axis=0
                    ),
                    dims=("klass", "covariate"),
                )
                # Logits vary per observation: (n_obs, K)
                logits = intercepts[None, :] + pt.dot(X_tensor, beta.T)
            else:
                # Fallback to static class probabilities if no covariates
                logits = intercepts[None, :]

            # 3. Compute observation-level and global class sizes
            pi_obs = pm.Deterministic(
                "class_sizes_obs",
                pt.special.softmax(logits, axis=1),
                dims=("obs", "klass"),
            )
            # Global class sizes are the average membership probability across the sample
            pi = pm.Deterministic("class_sizes", pt.mean(pi_obs, axis=0), dims="klass")
            for k in range(K):
                pm.Deterministic(f"class_size_{k + 1}", pi[k])

            # 4. Per-class item-endorsement probabilities θ (K x J)
            theta = pm.Beta(
                "item_prob",
                alpha=cfg.item_prior_a,
                beta=cfg.item_prior_b,
                shape=(K, J),
                dims=("klass", "item"),
            )
            pm.Deterministic("class_profile", theta, dims=("klass", "item"))

            # 5. Marginal mixture log-likelihood (class labels integrated out)
            Yt = pt.as_tensor_variable(Y)
            log_theta = pt.log(theta)
            log_1m = pt.log1p(-theta)
            comp = pt.dot(Yt, log_theta.T) + pt.dot(1.0 - Yt, log_1m.T)  # (n_obs, K)
            weighted = comp + pt.log(pi_obs)
            logp = pt.logsumexp(weighted, axis=1)  # (n_obs,)
            pm.Potential("lca_loglik", pt.sum(logp))

            # Posterior membership probabilities for reporting
            pm.Deterministic(
                "class_responsibility",
                pt.special.softmax(weighted, axis=1),
                dims=("obs", "klass"),
            )

        return model

    # -- estimands + reporting ----------------------------------------------

    def _default_estimands(self):
        K = self.model_params.n_classes
        return [
            latent_scalar(
                f"class_size_{k + 1}",
                var=f"class_size_{k + 1}",
                kind="class_size",
                units="proportion",
                causal_assumptions=f"Posterior share of the population in class C{k + 1}.",
            )
            for k in range(K)
        ]

    def class_profile_summary(self, hdi_prob: float = 0.94):
        """Per-(class, item) endorsement probability P(item=1 | class) — mean + HDI."""
        import arviz as az
        import pandas as pd

        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        prof = self._trace.posterior["class_profile"]
        mean = prof.mean(("chain", "draw")).values
        hdi = az.hdi(self._trace, var_names=["class_profile"], hdi_prob=hdi_prob)[
            "class_profile"
        ].values
        sizes = self._trace.posterior["class_sizes"].mean(("chain", "draw")).values
        rows = []
        for k in range(mean.shape[0]):
            for j, item in enumerate(self.item_names):
                rows.append(
                    {
                        "class": f"C{k + 1}",
                        "size": float(sizes[k]),
                        "item": item,
                        "prob": float(mean[k, j]),
                        "hdi_low": float(hdi[k, j, 0]),
                        "hdi_high": float(hdi[k, j, 1]),
                    }
                )
        return pd.DataFrame(rows)

    def covariate_effects_summary(self):
        """Returns the demographic effects (beta) on segment membership.
        Positive values mean higher values of the demographic increase the likelihood
        of belonging to that class relative to Class 1 (the reference class)."""
        import pandas as pd

        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if not self.has_covariates:
            return "No demographic covariates were provided to this model."

        beta = (
            self._trace.posterior["beta_cov"].mean(("chain", "draw")).values
        )  # (K, D)
        rows = []
        for k in range(self.model_params.n_classes):
            for d, cov in enumerate(self.covariate_names):
                rows.append(
                    {
                        "class": f"C{k + 1}",
                        "covariate": cov,
                        "effect_size": float(beta[k, d]),
                        "interpretation": (
                            "Reference Category (Fixed at 0)"
                            if k == 0
                            else (
                                f"Positive: Higher {cov} increases likelihood of C{k+1} vs C1"
                                if beta[k, d] > 0
                                else f"Negative: Higher {cov} decreases likelihood of C{k+1} vs C1"
                            )
                        ),
                    }
                )
        return pd.DataFrame(rows)


GARDEN_MODEL = BayesianLCR


def synthetic_lca_panel_with_demographics(n: int = 800, seed: int = 42):
    """Generates synthetic purchase data (6 brands) and demographic data (Age, Income).
    - Class 1 (35%): Older, higher income. Buys brands 1-3.
    - Class 2 (65%): Younger, lower income. Buys brands 4-6.
    """
    import pandas as pd
    from mmm_framework.config import (
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset

    rng = np.random.default_rng(seed)

    # Generate demographics (standardized)
    age = rng.normal(loc=0.0, scale=1.0, size=n)
    income = rng.normal(loc=0.0, scale=1.0, size=n)

    # True relationship: Age and Income drive class membership
    # Logit for Class 2 relative to Class 1:
    # Younger (negative age effect) and lower income (negative income effect) -> Class 2
    logits = 0.5 - 1.5 * age - 1.2 * income
    prob_class2 = 1 / (1 + np.exp(-logits))
    z = (rng.random(n) < prob_class2).astype(int)  # 0 = Class 1, 1 = Class 2

    profiles = np.array(
        [
            [0.85, 0.85, 0.85, 0.15, 0.15, 0.15],  # Class 1 (Premium)
            [0.15, 0.15, 0.15, 0.85, 0.85, 0.85],  # Class 2 (Value)
        ]
    )

    Y = (rng.random((n, 6)) < profiles[z]).astype(int)
    cols = [f"brand_{j + 1}" for j in range(6)]

    df_purchases = pd.DataFrame(Y, columns=cols)
    df_demographics = pd.DataFrame({"Age": age, "Income": income})

    periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    media = cols[1:]

    config = MFFConfig(
        kpi=KPIConfig(name=cols[0], dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD]) for c in media
        ],
        controls=[],
    )

    panel = PanelDataset(
        y=df_purchases[cols[0]],
        X_media=df_purchases[media],
        X_controls=df_demographics,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=media,
            controls=["Age", "Income"],
        ),
        index=periods,
        config=config,
    )
    return panel


if __name__ == "__main__":
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    panel = synthetic_lca_panel_with_demographics()
    print("Fitting Bayesian Latent Class Regression (MAP) with Demographics...")

    mmm = BayesianLCR(
        panel,
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params={"n_classes": 2},
    )
    mmm.fit(method="map", random_seed=42)

    print("\n=== Recovered Class Profiles (Brand Penetration) ===")
    summary = mmm.class_profile_summary()
    print(
        summary.pivot(index="item", columns="class", values="prob").round(2).to_string()
    )

    print("\n=== Demographic Effects on Segment Membership ===")
    print("Note: Class 1 (C1) is the reference category.")
    effects = mmm.covariate_effects_summary()
    print(effects.to_string(index=False))
