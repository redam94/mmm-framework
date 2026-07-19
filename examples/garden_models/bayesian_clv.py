"""Bayesian CLV (Model Garden example) — transaction-level customer lifetime
value as a **non-MMM** garden family: BG/NBD purchase/dropout dynamics +
Gamma-Gamma monetary value, from per-customer RFM summaries.

There are no channels, no spend, no time-series KPI — each "observation" is a
CUSTOMER (frequency, recency, T, monetary). It answers:

    "How many more purchases will each customer make, what is each future
     purchase worth, who is still alive — and therefore what is the expected
     lifetime value of the book of customers?"

How it plugs in (the non-MMM contract)
--------------------------------------
Subclasses :class:`~mmm_framework.garden.CustomMMM` with
``__garden_model_kind__ = "clv"`` (exempt from channel/beta gates), overriding
only ``_prepare_data`` (read the RFM columns) and ``_build_model`` (the
BG/NBD + Gamma-Gamma graph from :mod:`mmm_framework.ltv.likelihood`). ``fit`` /
serialization / the estimand engine are inherited. Quantities of interest are
scalar deterministics surfaced via ``DEFAULT_ESTIMANDS`` (mean/total CLV, mean
expected purchases, mean P(alive)) plus a ``customer_value_summary()`` table.

Model
-----
Population heterogeneity (all customers share these):

* purchase rate  ``lam_i ~ Gamma(r, alpha)`` — ``r``/``alpha`` estimated,
* dropout        ``p_i ~ Beta(a, b)`` after every purchase,
* spend scale    ``nu_i ~ Gamma(q, gamma)``; values ``z ~ Gamma(p, nu_i)``.

The per-customer latents are INTEGRATED OUT (closed forms — NUTS-able, no
per-customer parameters), so the graph has just 7 free scalars regardless of
the number of customers.

CLV = E[purchases over ``horizon_periods``] x E[value per purchase], with a
mid-horizon discount ``(1+weekly rate)^(-horizon/2)`` (MVP approximation).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field

from mmm_framework.estimands.registry import latent_scalar
from mmm_framework.garden import CustomMMM
from mmm_framework.ltv.likelihood import (
    bgnbd_expected_purchases,
    bgnbd_loglik,
    bgnbd_p_alive,
    gamma_gamma_expected_value,
    gamma_gamma_loglik,
)

_RFM_DEFAULT = {
    "frequency": "frequency",
    "recency": "recency",
    "T": "T",
    "monetary": "monetary",
}


class CLVConfig(BaseModel):
    """Bespoke, settable configuration for :class:`BayesianCLV` (CONFIG_SCHEMA)."""

    #: Future horizon (in the RFM time unit, typically weeks) the CLV covers.
    horizon_periods: int = Field(default=52, ge=1)
    #: Annual financial discount rate applied at mid-horizon (MVP approximation).
    discount_rate_annual: float = Field(default=0.0, ge=0.0)
    #: Fit the Gamma-Gamma monetary block (off -> CLV is expected PURCHASES).
    monetary_model: bool = True
    #: Column-name mapping onto the dataset's observed columns.
    rfm_columns: dict[str, str] | None = None
    #: Acquisition-segment support (Phase 5): the numeric CODE column in the
    #: dataset (``rfm_panel`` emits ``segment_code``) + the code→label list.
    #: Set both to get per-segment ``segment_clv_<label>`` deterministics —
    #: the posterior CLV per acquisition channel that values acquisition
    #: experiments on LIFETIME value (see ``segment_model_params``).
    segment_column: str | None = None
    segment_labels: list[str] | None = None
    #: Prior scales (HalfNormal sigmas) for the BG/NBD population params.
    prior_sigma_r: float = Field(default=2.0, gt=0)
    prior_sigma_alpha: float = Field(default=20.0, gt=0)
    prior_sigma_ab: float = Field(default=3.0, gt=0)
    #: Gamma-Gamma prior scales. NB ``q`` must stay away from 1 for the
    #: population mean ``p*gamma/(q-1)`` — its prior is a shifted HalfNormal.
    prior_sigma_p: float = Field(default=10.0, gt=0)
    prior_sigma_q: float = Field(default=5.0, gt=0)
    prior_sigma_gamma: float = Field(default=30.0, gt=0)

    model_config = {"extra": "forbid"}


class BayesianCLV(CustomMMM):
    """BG/NBD + Gamma-Gamma customer lifetime value as a garden model."""

    __garden_model_kind__ = "clv"
    CONFIG_SCHEMA = CLVConfig
    DEFAULT_ESTIMANDS = [
        latent_scalar(
            "mean_clv",
            var="mean_clv",
            kind="clv",
            units="value/customer",
            causal_assumptions="Expected discounted value per customer over the horizon, "
            "assuming the fitted purchase/dropout/value process is stationary.",
        ),
        latent_scalar("total_clv", var="total_clv", kind="clv", units="value"),
        latent_scalar(
            "mean_expected_purchases",
            var="mean_expected_purchases",
            kind="purchases",
            units="purchases/customer",
        ),
        latent_scalar(
            "mean_p_alive", var="mean_p_alive", kind="retention", units="probability"
        ),
    ]

    # -- data ----------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Pull the RFM arrays off the observed columns (by-name mapping via
        ``model_params.rfm_columns``), then fill the non-MMM defaults."""
        observed = self.dataset.observed()
        cols = dict(_RFM_DEFAULT)
        if self.model_params.rfm_columns:
            cols.update(self.model_params.rfm_columns)
        missing = [
            v
            for k, v in cols.items()
            if v not in observed.columns and not (k == "monetary")
        ]
        if missing:
            raise ValueError(
                f"RFM columns {missing} not found in the dataset (have "
                f"{list(observed.columns)}); set model_params.rfm_columns."
            )
        self.rfm_x = observed[cols["frequency"]].to_numpy(dtype=np.float64)
        self.rfm_t_x = observed[cols["recency"]].to_numpy(dtype=np.float64)
        self.rfm_T = observed[cols["T"]].to_numpy(dtype=np.float64)
        has_monetary = cols["monetary"] in observed.columns
        self.rfm_m = (
            observed[cols["monetary"]].to_numpy(dtype=np.float64)
            if has_monetary
            else None
        )
        self.use_monetary = bool(self.model_params.monetary_model and has_monetary)

        # acquisition segments (Phase 5): numeric codes + labels
        self.segment_codes = None
        self.segment_labels: list[str] = []
        seg_col = self.model_params.segment_column
        if seg_col:
            if seg_col not in observed.columns:
                raise ValueError(
                    f"segment_column '{seg_col}' not in the dataset (have "
                    f"{list(observed.columns)})."
                )
            codes = observed[seg_col].to_numpy(dtype=np.int64)
            labels = self.model_params.segment_labels or [
                f"seg{int(c)}" for c in np.unique(codes)
            ]
            if codes.min() < 0 or codes.max() >= len(labels):
                raise ValueError(
                    f"segment codes must index segment_labels (0..{len(labels) - 1});"
                    f" got range [{codes.min()}, {codes.max()}]."
                )
            self.segment_codes = codes
            self.segment_labels = [str(lbl) for lbl in labels]

        if (
            (self.rfm_x < 0).any()
            or (self.rfm_t_x < 0).any()
            or (self.rfm_T <= 0).any()
        ):
            raise ValueError("RFM needs frequency>=0, recency>=0 and T>0 per customer.")
        if ((self.rfm_x > 0) & (self.rfm_t_x <= 0)).any():
            raise ValueError("repeat buyers (frequency>0) must have recency>0.")

        self.n_obs = len(self.rfm_x)
        self._set_non_mmm_defaults()

    # -- model ---------------------------------------------------------------

    def _build_model(self) -> pm.Model:
        cfg = self.model_params
        x, t_x, T = self.rfm_x, self.rfm_t_x, self.rfm_T
        horizon = float(cfg.horizon_periods)
        coords = {"customer": np.arange(self.n_obs)}

        with pm.Model(coords=coords) as model:
            # BG/NBD population parameters (7 free scalars total).
            r = pm.HalfNormal("r", sigma=cfg.prior_sigma_r)
            alpha = pm.HalfNormal("alpha", sigma=cfg.prior_sigma_alpha)
            a = pm.HalfNormal("a", sigma=cfg.prior_sigma_ab)
            b = pm.HalfNormal("b", sigma=cfg.prior_sigma_ab)
            pm.Potential("bgnbd_ll", pt.sum(bgnbd_loglik(r, alpha, a, b, x, t_x, T)))

            p_alive = pm.Deterministic(
                "p_alive", bgnbd_p_alive(r, alpha, a, b, x, t_x, T), dims="customer"
            )
            expected_purchases = pm.Deterministic(
                "expected_purchases",
                bgnbd_expected_purchases(r, alpha, a, b, x, t_x, T, horizon),
                dims="customer",
            )
            pm.Deterministic("mean_p_alive", pt.mean(p_alive))
            pm.Deterministic("mean_expected_purchases", pt.mean(expected_purchases))

            if self.use_monetary:
                p_gg = pm.HalfNormal("p_gg", sigma=cfg.prior_sigma_p)
                # shifted: q > 1 by construction -> the population mean
                # p*gamma/(q-1) always exists (no prior mass at the pole).
                q_raw = pm.HalfNormal("q_raw", sigma=cfg.prior_sigma_q)
                q_gg = pm.Deterministic("q_gg", 1.0 + q_raw)
                gamma_gg = pm.HalfNormal("gamma_gg", sigma=cfg.prior_sigma_gamma)
                pm.Potential(
                    "gg_ll",
                    pt.sum(gamma_gamma_loglik(p_gg, q_gg, gamma_gg, x, self.rfm_m)),
                )
                expected_value = pm.Deterministic(
                    "expected_avg_value",
                    gamma_gamma_expected_value(p_gg, q_gg, gamma_gg, x, self.rfm_m),
                    dims="customer",
                )
            else:
                expected_value = pt.ones_like(expected_purchases)

            # CLV with a mid-horizon discount (MVP; per-period refinement later).
            weekly = (1.0 + cfg.discount_rate_annual) ** (1.0 / 52.0) - 1.0
            disc = (1.0 + weekly) ** (-horizon / 2.0)
            clv = pm.Deterministic(
                "clv", expected_purchases * expected_value * disc, dims="customer"
            )
            pm.Deterministic("mean_clv", pt.mean(clv))
            pm.Deterministic("total_clv", pt.sum(clv))

            # Per-acquisition-segment CLV (Phase 5): the posterior mean lifetime
            # value of a customer ACQUIRED BY each channel — the number an
            # acquisition experiment should value a conversion at.
            if self.segment_codes is not None:
                for k, label in enumerate(self.segment_labels):
                    mask = self.segment_codes == k
                    if not mask.any():
                        continue
                    pm.Deterministic(
                        f"segment_clv_{label}", pt.mean(clv[np.flatnonzero(mask)])
                    )
                    pm.Deterministic(
                        f"segment_p_alive_{label}",
                        pt.mean(p_alive[np.flatnonzero(mask)]),
                    )

        return model

    # -- estimands ------------------------------------------------------------

    def _default_estimands(self):
        """Static CLV estimands + one ``segment_clv_<label>`` per acquisition
        segment (dynamic in the segment labels, so declared here)."""
        out = list(self.DEFAULT_ESTIMANDS)
        for label in self.segment_labels:
            out.append(
                latent_scalar(
                    f"segment_clv_{label}",
                    var=f"segment_clv_{label}",
                    kind="clv",
                    units="value/customer",
                    causal_assumptions=(
                        f"Mean CLV of customers acquired by '{label}' — the "
                        "value_per_conversion for an acquisition experiment on "
                        "that channel."
                    ),
                )
            )
        return out

    # -- reporting ------------------------------------------------------------

    def customer_value_summary(self, hdi_prob: float = 0.94):
        """Population-level value table: mean CLV / P(alive) / expected
        purchases (+ HDIs) and the CLV decile profile across customers — the
        non-MMM latent-structure table (analogue of CFA loadings / LCA
        profiles), rendered by the report's latent section."""
        import pandas as pd

        from mmm_framework.utils.arviz_compat import hdi_dataset

        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        post = self._trace.posterior
        rows = []
        for var, label in (
            ("mean_clv", "Mean CLV per customer"),
            ("total_clv", "Total book CLV"),
            ("mean_expected_purchases", "Mean expected purchases"),
            ("mean_p_alive", "Mean P(alive)"),
        ):
            if var not in post:
                continue
            hdi = hdi_dataset(self._trace, hdi_prob, var_names=[var])[var].values
            rows.append(
                {
                    "quantity": label,
                    "mean": float(post[var].mean(("chain", "draw")).values),
                    "hdi_low": float(np.asarray(hdi).reshape(-1)[0]),
                    "hdi_high": float(np.asarray(hdi).reshape(-1)[1]),
                }
            )
        clv_mean = post["clv"].mean(("chain", "draw")).values  # (n_customers,)
        for dec in (0.5, 0.8, 0.9, 0.99):
            rows.append(
                {
                    "quantity": f"CLV p{int(dec * 100)} customer",
                    "mean": float(np.quantile(clv_mean, dec)),
                    "hdi_low": float("nan"),
                    "hdi_high": float("nan"),
                }
            )
        # per-acquisition-segment CLV (when segments were configured)
        for label in self.segment_labels:
            var = f"segment_clv_{label}"
            if var not in post:
                continue
            hdi = hdi_dataset(self._trace, hdi_prob, var_names=[var])[var].values
            rows.append(
                {
                    "quantity": f"Segment CLV — {label}",
                    "mean": float(post[var].mean(("chain", "draw")).values),
                    "hdi_low": float(np.asarray(hdi).reshape(-1)[0]),
                    "hdi_high": float(np.asarray(hdi).reshape(-1)[1]),
                }
            )
        return pd.DataFrame(rows)

    def segment_clv_means(self) -> dict[str, float]:
        """Posterior-mean CLV per acquisition segment — the input to
        :func:`mmm_framework.ltv.clv_to_cac` and the per-channel
        ``value_per_conversion`` for acquisition experiments."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        post = self._trace.posterior
        out = {}
        for label in self.segment_labels:
            var = f"segment_clv_{label}"
            if var in post:
                out[label] = float(post[var].mean(("chain", "draw")).values)
        return out


GARDEN_MODEL = BayesianCLV


def segment_model_params(rfm) -> dict:
    """The ``model_params`` entries that switch on per-segment CLV for an RFM
    frame carrying a ``segment`` column (``transactions_to_rfm(segment_col=…)``):
    ``rfm_panel`` encodes the segment as the numeric ``segment_code`` column and
    this returns the matching ``{"segment_column", "segment_labels"}``."""
    if "segment" not in rfm.columns:
        return {}
    labels = sorted(rfm["segment"].astype(str).unique())
    return {"segment_column": "segment_code", "segment_labels": labels}


def rfm_panel(rfm):
    """Wrap an RFM frame (``transactions_to_rfm`` output) as the
    :class:`PanelDataset` the garden constructor takes. Each "period" indexes a
    customer (the non-MMM convention, mirroring the LCA example); frequency is
    listed as the KPI and recency/T/monetary as measured columns. A ``segment``
    column is factorized into a numeric ``segment_code`` column (pair with
    ``segment_model_params`` to enable per-segment CLV)."""
    import pandas as pd

    from mmm_framework.config import (
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset

    rfm = rfm.copy()
    if "segment" in rfm.columns:
        labels = sorted(rfm["segment"].astype(str).unique())
        rfm["segment_code"] = (
            rfm["segment"].astype(str).map({s: i for i, s in enumerate(labels)})
        )
    cols = ["recency", "T"] + (["monetary"] if "monetary" in rfm.columns else [])
    if "segment_code" in rfm.columns:
        cols.append("segment_code")
    periods = pd.date_range("2021-01-04", periods=len(rfm), freq="W-MON")
    config = MFFConfig(
        kpi=KPIConfig(name="frequency", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD]) for c in cols
        ],
        controls=[],
    )
    return PanelDataset(
        y=pd.Series(
            rfm["frequency"].to_numpy(dtype=float), index=periods, name="frequency"
        ),
        X_media=pd.DataFrame(
            {c: rfm[c].to_numpy(dtype=float) for c in cols}, index=periods
        ),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=cols,
            controls=None,
        ),
        index=periods,
        config=config,
    )


if __name__ == "__main__":
    from mmm_framework.config import ModelConfig
    from mmm_framework.ltv import transactions_to_rfm
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType
    from mmm_framework.synth.dgp_clv import make_clv_world

    world = make_clv_world(seed=7, n_customers=2000)
    rfm = transactions_to_rfm(
        world.transactions, value_col="value", observation_end=world.observation_end
    )
    print(f"Fitting BayesianCLV (MAP) on {len(rfm)} synthetic customers…")
    mmm = BayesianCLV(
        rfm_panel(rfm),
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params={"horizon_periods": world.truth["holdout_weeks"]},
    )
    mmm.fit(method="map", random_seed=7)
    post = mmm._trace.posterior
    print("\nRecovered vs true population parameters:")
    for name, key in (
        ("r", "r"),
        ("alpha", "alpha"),
        ("a", "a"),
        ("b", "b"),
        ("p_gg", "p_gg"),
        ("q_gg", "q_gg"),
        ("gamma_gg", "gamma_gg"),
    ):
        est = float(post[name].mean().values) if name in post else float("nan")
        print(f"  {name:9s} est {est:7.3f}   true {world.truth[key]:7.3f}")
    print("\n", mmm.customer_value_summary().to_string(index=False))
